#!/bin/bash

#!/bin/bash

echo "开始快速验证训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 快速验证训练命令
python -m train.train_anytop \
  --model_prefix quick_test_0818 \
  --objects_subset bipeds \
  --lambda_geo 1.0 \
  --balanced \
  --gen_during_training \
  --overwrite \
  --batch_size 16 \
  --latent_dim 128 \
  --diffusion_steps 100 \
  --num_frames 120 \
  --temporal_window 31 \
  --lr 1e-4 \
  --num_steps 1000 \
  --save_interval 500 \
  --log_interval 100 \
  --ml_platform_type WandBPlatform

echo "快速验证训练完成！"