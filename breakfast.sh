python -m src.train \
    --decode_sample_rate 30 --window_size 21 --hidden_size 64 --lr_decay_iter 60000 \
    --gpu 0 --lr 0.01 --epoch 100000 \
    --space_size 3 --pred_size 0 --autoencoder_weight 0.35 \
    --exp "run1" --data breakfast --split 1 \
    --ali_every 5000 --print_every 1000 --seg_every 10000 \
    --resume 'max'
