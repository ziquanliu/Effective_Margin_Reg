#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0

lambda=3.0
temp=40.0
python main-AT-EMR.py --lambda_EMR $lambda --EMR_softmax_temp $temp --data_root '/home/data/cifar-10' --model_root './valid_v2_2000_temp_lambda_EMR_'$lambda'_decay_LGR_temp_'$temp'_wd_5e-4_epoch_100_v2' -w 0.0005 -e 0.0314 -p 'linf' --adv_train --affix 'linf' --log_root 'log_valid_v2_2000_temp_lambda_EMR_'$lambda'_decay_LGR_temp_'$temp'_wd_5e-4_epoch_100_v2' --gpu '0' -m_e 100
