import os
import sys
from typing import List #  In order to specify the types of parameters and return values by List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000, # validation set which measns it will take such numbers data for test(validation验证集)
    # the following some args will be influnced by this paramater 'evaluation_strategy', 'eval_steps', 'load_best_model_at_end' in trainer.args
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ], # define lora_target_modules as a list type and string type in such list
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: #  if LOCAL_RANK is NOT exit, return 0 
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # 环境变量WORLD_SIZE不存在时返回1
    # os.environ.get()是python中os模块获取环境变量的一个方法,如果有这个键,返回对应的值,如果没有,则返回none
    ddp = world_size != 1  # if world_size == 1, ddp=false; if world_size != 1, ddp=true
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # device_map will be a dict and '' will be key, int() or 0 will be value
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project # os.environ['环境变量名称']='新环境变量值' #其中key和value均为string类型
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16, # data type of the quantized Tensor 量化张量的数据类型 the original torch is float32 and we learn from the idea from quantization to reduce model inference time
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        # chechpoint_name will be 'resume_from_checkpoint'\pythorch_model.bin
        # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)  # torch.load(model_path) for load model, and model_path should be the model fiel path
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # torch.cuda.device_count() is used to determine the number of GPUS
        # 'not' has higher priority than 'and', so ddp==false and torch.cuda.device_count() is more than 1 will run the following code
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer( # must be used on GPU, Trainer will be much more slower on CPU
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,  # per_device_train_batch_size（：obj：`int`，`optional`，defaults to 8）： The batch size per GPU/TPU core/CPU for training 每个GPU / TPU内核/ CPU的批处理大小
            gradient_accumulation_steps=gradient_accumulation_steps, # gradient_accumulation_steps: (:obj:`int`, `optional`, defaults to 1): Number of updates steps to accumulate the gradients for, before performing a backward/update pass.在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
            warmup_steps=100, # warmup_steps (:obj:`int`, `optional`, defaults to 0):Number of steps used for a linear warmup from 0 to :obj:`learning_rate`.线性预热所用的步数（从0到：learning_rate）。
            num_train_epochs=num_epochs, # num_train_epochs(:obj:`float`, `optional`, defaults to 3.0): Total number of training epochs to perform.要执行的训练轮数总数
            learning_rate=learning_rate, # learning_rate (:obj:`float`, `optional`, defaults to 5e-5):The initial learning rate for Adam.Adam初始学习率。#这里不知道为什么强调Adam？
            fp16=True,# fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.是否使用16位混合精度训练（通过NVIDIA apex）而不是32位训练
            
            # logging_steps=10, every 10 steps will have an loss number
            logging_steps=1, # logging_steps (:obj:`int`, `optional`, defaults to 500):Number of update steps between two logs.两个日志记录之间的更新步骤数。
            optim="adamw_torch", # 不是TrainingArguments里的默认参数 mean use adamw optimization(kind of 优化器)
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # evaluation_strategy (:obj:`str` or :class:`~transformers.trainer_utils.EvaluationStrategy`, `optional`, defaults to :obj:`"no"`):
            # The evaluation strategy to adopt during training. Possible values are: 评估策略
            #  :obj:`"no"`: No evaluation is done during training.训练期间不进行评估
            #  :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.每个 :obj:`eval_steps` 都会进行评估（并记录）
            #  :obj:`"epoch"`: Evaluation is done at the end of each epoch.评估在每个 epoch 结束时完成
            save_strategy="steps", # 不是TrainingArguments里的默认参数 mean every steps will be save?
            eval_steps=200 if val_set_size > 0 else None,# eval_steps (:obj:`int`, `optional`):Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`. Will default to the same value as :obj:`logging_steps` if not set.如果:obj:`evaluation_strategy="steps"`，则两次评估之间的更新步骤数
            save_steps=1,# save_steps (:obj:`int`, `optional`, defaults to 500):Number of updates steps before two checkpoint saves.
            # save_steps=200, which means that it will save every 200 steps as result(checkpoint)
            output_dir=output_dir, # output_dir (:obj:`str`):The output directory where the model predictions and checkpoints will be written.模型预测和检查点的输出目录。必须声明的字段
            save_total_limit=3, # save_total_limit (:obj:`int`, `optional`):If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in :obj:`output_dir`.如果设置具体数值，将限制checkpoints的总数并覆盖旧的checkpoints
            # save_total_limit='x' represent that there will save the latest 'x' checkpoint 
            load_best_model_at_end=True if val_set_size > 0 else False, # load_best_model_at_end (:obj:`bool`, `optional`, defaults to :obj:`False`):Whether or not to load the best model found during training at the end of training.是否在训练结束时加载训练期间找到的最佳模型
            ddp_find_unused_parameters=False if ddp else None, # ddp_find_unused_parameters (:obj:`bool`, `optional`):When using distributed training, the value of the flag :obj:`find_unused_parameters` passed to:obj:`DistributedDataParallel`. Will default to :obj:`False` if gradient checkpointing is used, :obj:`True`otherwise.
            # 使用分布式训练时，:obj:`find_unused_pa​​rameters` 的值传递给:obj:`DistributedDataParallel`。如果使用梯度checkpointing，则默认为 :obj:`False`，否则为`True`
            group_by_length=group_by_length, # group_by_length (:obj:`bool`, `optional`, defaults to :obj:`False`):Whether or not to group together samples of roughly the same legnth in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.是否将训练数据集中长度大致相同的样本分组在一起（以最小化应用填充并提高效率）。仅在应用动态填充时有用
            report_to="wandb" if use_wandb else None, # 用于报告结果和日志的集成列表 use wandb
            run_name=wandb_run_name if use_wandb else None, # run_name (:obj:`str`, `optional`):A descriptor for the run. Typically used for `wandb <https://www.wandb.com/>`_ logging.
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":  # determine torch version and system whether or not win32, in general we use torchversion more than 2
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint) # used trainer(transformers.Trainer).train to run

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
# 定义若干函数，使用 fire.Fire()
# 方便修改调用的函数中的默认参数的值
# 理解成点火，把定义的函数都暴露出来运行
