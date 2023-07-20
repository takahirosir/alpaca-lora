python finetune.py \
    --base_model 'models/vicuna-7b-v1.1' \
    --data_path 'trans_chinese_alpaca_data.json' \
    --output_dir './lora-alpaca-zh' \
    --val_set_size 1