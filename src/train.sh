#!/bin/bash
# fully supervised training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 train.py --device_ids 0 1 2 3 --num_workers 8 --batch_size 12 --folds 0 1 2 3 --problem_type parts --lr 3e-5 --model UNet16 --jaccard_weight 0.3 --max_epochs 100 --model_save_dir ../model_ckpt --input_height 512 --input_width 640 --tb_log True

# fully supervised + load model ckpt
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3.6 train.py --ckpt_dir ../model_ckpt/UNet11_binary --device_ids 0 1 --num_workers 8 --batch_size 8 --folds 0 1 2 3 --problem_type binary --lr 1e-5 --model UNet11 --jaccard_weight 0.3 --max_epochs 10 --model_save_dir ../model_ckpt --input_height 512 --input_width 640

# semi-supervised training
CUDA_VISIBLE_DEVICES=2,3 python3.6 train.py --device_ids 0 1 --num_workers 8 --batch_size 4 --folds 0 1 2 3 --problem_type parts --lr 1e-5 --model TAPNet11 --jaccard_weight 0.3 --max_epochs 100 --model_save_dir ../model_ckpt --input_height 512 --input_width 640 --semi True --semi_method rev_flow --semi_percentage 0.5 --semi_loss_alpha 1e-2 --tb_log True

# # fully supervised training (256, 512)
# CUDA_VISIBLE_DEVICES=2,3 python3.6 train.py --device_ids 0 1 --num_workers 8 --batch_size 6 --folds 0 1 2 3 --problem_type parts --lr 3e-5 --model UNet11 --jaccard_weight 0.3 --max_epochs 100 --model_save_dir ../model_ckpt --input_height 256 --input_width 320 --tb_log True
