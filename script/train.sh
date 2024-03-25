python src/train.py \
    --input_file data/sample.json \
    --encoder_name xlm-roberta-large \
    --label_coff 1 \
    --rationale_coff 15 \
    --lr 5e-5 \
    --epoch 3\
    --frac_warmup 0.1 \
    --gradient_accumulations 8 \
    --batch_size 2 