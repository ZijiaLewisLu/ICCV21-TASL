
python -m src.train \
    --decode_sample_rate 30 --window_size 21 --hidden_size 64 --lr_decay_iter 60000 \
    --gpu 0 --lr 0.01 --epoch 200000 \
    --space_size 10 --pred_size 3 --autoencoder_weight 0.2 \
    --exp "run1" --edge_window 6 --edge_step 2 \
    --data hollywood --split 1 \
    --ali_every 5000 --print_every 1000 --seg_every 10000 --resume max

