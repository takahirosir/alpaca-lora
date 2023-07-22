### Local Setup
1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
### Download Model
1. Download llama-7b
    ```bash
    bash scripts/llama-7b.sh
    ```
2. Download other model such like vicuna or alpaca please go to hf e.g.
    ```bash
    bash scripts/alpaca-lora-7b.sh
    ```
### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:
1. use sh file
    ```bash
    bash scripts/overfit.sh
    ```
2. ues python command
    ```bash
    python finetune.py \
    --base_model 'model/llama-7b-hf' \
    --data_path 'overfitting' \
    --output_dir './output'
    ```
3. We can also tweak our hyperparameters:
    ```bash
    python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
    ```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:
1. use sh file
    ```bash
    bash scripts/generate.sh
    ```
2. ues python command
    ```bash
    python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
    ```

### Official weights

The most recent "official" Alpaca-LoRA adapter available at [`tloen/alpaca-lora-7b`](https://huggingface.co/tloen/alpaca-lora-7b) was trained on March 26 with the following command:

```bash
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```