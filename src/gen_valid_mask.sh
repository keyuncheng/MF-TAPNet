#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 gen_valid_mask.py --ckpt_dir ../model_ckpt --mask_save_dir ../valid_masks --device_ids 0 1 2 3 --num_workers 8 --batch_size 12 --folds 0 1 2 3 --problem_type binary --model UNet11 --input_height 512 --input_width 640 --pad False
