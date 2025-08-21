#!/bin/bash

# 训练SMPL Human模型的脚本

echo "开始训练SMPL Human模型..."

# 设置WandB环境变量
export WANDB_PROJECT="anytop_smpl_human"
export WANDB_NAME="smpl_human_training"
export WANDB_ENTITY="lingfanb-university-college-london-ucl-"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练命令
python -m train.train_anytop \
  --model_prefix smpl_human \
  --objects_subset bipeds \
  --lambda_geo 1.0 \
  --balanced \
  --gen_during_training \
  --overwrite \
  --batch_size 8 \
  --latent_dim 128 \
  --diffusion_steps 100 \
  --num_frames 120 \
  --temporal_window 31 \
  --lr 1e-4 \
  --num_steps 100000 \
  --save_interval 10000 \
  --log_interval 50 \
  --ml_platform_type WandBPlatform

echo "训练完成！"
