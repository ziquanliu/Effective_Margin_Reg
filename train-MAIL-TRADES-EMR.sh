#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0

python main-TRADES-MAIL_EMR.py --lambda_EMR 1.0 --data_root '/home/data/cifar-10' --model_root './MAIL_TRADES_5.0_slope_5_bias_0.05_lambda_EMR_1.0_decay_LGR_v2' -e 0.0314 -k 10 -p 'linf' --adv_train --affix 'linf' --log_root 'log_MAIL_TRADES_5.0_slope_5_bias_0.05_lambda_EMR_1.0_decay_LGR_v2' --gpu '0' --max_epoch 100
