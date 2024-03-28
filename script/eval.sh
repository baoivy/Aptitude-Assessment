python src/predict.py \
    --input_file data/sample.json \
    --output_file output/result.jsonl \
    --checkpoint_path /content/drive/MyDrive/AISIA-Assignment/lightning_logs/version_24/checkpoints/epoch=0-step=576.ckpt \
    --model xlm-roberta-large \
    --predict True
