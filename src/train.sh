#!/bin/bash
# fully supervised training
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 train.py --device_ids 0 1 2 3 --num_workers 8 --batch_size 12 --folds 0 1 2 3 --problem_type parts --lr 3e-5 --lr_decay 0.9 --lr_decay_epochs 10 --weight_decay 1e-4 --model UNet16 --jaccard_weight 0.3 --max_epochs 100 --model_save_dir ../model_ckpt --input_height 512 --input_width 640

# fully supervised + load model ckpt
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3.6 train.py --ckpt_dir ../model_ckpt/UNet11_binary --device_ids 0 1 --num_workers 8 --batch_size 8 --folds 0 1 2 3 --problem_type binary --jaccard_weight 0.3 --max_epochs 10 --lr 1e-5 --lr_decay 0.9 --lr_decay_epochs 5 --weight_decay 1e-4 --model UNet11 --model_save_dir ../model_ckpt --input_height 512 --input_width 640

# semi-supervised training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 train.py --device_ids 0 1 2 3 --num_workers 8 --batch_size 8 --folds 0 1 2 3 --problem_type binary --jaccard_weight 0.3 --max_epochs 100 --lr 1e-5 --lr_decay 0.9 --lr_decay_epochs 5 --weight_decay 1e-4 --input_height 512 --input_width 640 --model UNet11 --model_save_dir ../model_ckpt --semi True --semi_method aug_gt --semi_percentage 0.5

