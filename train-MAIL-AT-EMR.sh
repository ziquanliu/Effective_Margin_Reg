#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0

python main-AT-MAIL-EMR.py --lambda_EMR 1.0 --beta_trades 6.0 --data_root '/home/data/cifar-10' --model_root './MAIL_AT_slope_30_bias_m0.07_lambda_EMR_1.0_decay_LGR_v2' -e 0.0314 -k 10 -p 'linf' --adv_train --affix 'linf' --log_root 'log_MAIL_AT_slope_30_bias_m0.07_lambda_EMR_1.0_decay_LGR_v2' --gpu '0' --max_epoch 100
