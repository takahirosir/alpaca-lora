python finetune.py \
    --base_model 'models/llama-7b' \
    --data_path 'overfitting.json' \
    --output_dir './overfit' \
    --num_epochs 10 \
    --val_set_size 0