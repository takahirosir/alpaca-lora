python finetune.py \
    --base_model 'models/vicuna-7b-v1.1' \
    --data_path 'alpaca_data_gpt4.json' \
    --output_dir './lora-alpaca' \
    --num_epochs 1