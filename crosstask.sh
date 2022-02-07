python -m src.train \
    --decode_sample_rate 4 --window_size 3 --hidden_size 64 --lr_decay_iter 60000 \
    --gpu 0 --lr 0.005 --epoch 100000 --exp "run1" \
    --space_size 3 --pred_size 0 --autoencoder_weight 0.35 \
    --edge_window 4 --edge_step 2 --infer_ew 6 --infer_es 1 \
    --data crosstask --split 1 \
    --hie_grammar dataset/CrossTask/transcript_20cluster.json \
    --ali_every 2500 --print_every 1000 --seg_every 5000 --resume max 

