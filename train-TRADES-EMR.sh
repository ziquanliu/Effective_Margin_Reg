#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0

lambda=1.0
temp=1.0
python main-TRADES-EMR.py --lambda_EMR $lambda --EMR_softmax_temp $temp --beta_trades 12.0 --data_root '/home/data/cifar-10' --model_root './valid_v2_2000_temp_TRADES_12.0_lambda_EMR_'$lambda'_decay_LGR_temp_'$temp'_wd_1e-3_v2' -e 0.0314 -p 'linf' -w 0.001 --adv_train --affix 'linf' --log_root 'log_valid_v2_2000_temp_TRADES_12.0_lambda_EMR_'$lambda'_decay_LGR_temp_'$temp'_wd_1e-3_v2' --gpu '0'
