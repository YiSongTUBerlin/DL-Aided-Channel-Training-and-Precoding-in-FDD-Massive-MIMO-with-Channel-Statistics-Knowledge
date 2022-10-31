#!/bin/sh

for B in 1 4 8 16 25 40
do
  python ACS_WEI_YU_complex/FDD_mmwave_precoding_torch.py --n_epochs 5000 --batch_size 1024 --lr 0.0001 --B $B --snr_dl 20 --beta_tr 8 --M 64 --K 6 --Lp_min 20 --Lp_max 20 --sample_num 5 --batch_num_per_epoch 20 --val_size 1000 --test_size 1000 --h_num 1 --max_theta 60
  python ACS_WEI_YU_complex/FDD_mmwave_precoding_torch.py --n_epochs 5000 --batch_size 1024 --lr 0.0001 --B $B --snr_dl 20 --beta_tr 8 --M 64 --K 6 --Lp_min 2 --Lp_max 2 --sample_num 5 --batch_num_per_epoch 20 --val_size 1000 --test_size 1000 --h_num 1 --max_theta 60
  python ACS_only_training_pilot_true/FDD_mmwave_precoding_torch.py --n_epochs 500 --batch_size 1024 --lr 0.0001 --B $B --snr_dl 20 --beta_tr 8 --M 64 --K 6 --Lp_min 20 --Lp_max 20 --sample_num 5 --batch_num_per_epoch 20 --val_size 1000 --test_size 1000 --h_num 1 --max_theta 60
  python ACS_only_training_pilot_true/FDD_mmwave_precoding_torch.py --n_epochs 500 --batch_size 1024 --lr 0.0001 --B $B --snr_dl 20 --beta_tr 8 --M 64 --K 6 --Lp_min 2 --Lp_max 2 --sample_num 5 --batch_num_per_epoch 20 --val_size 1000 --test_size 1000 --h_num 1 --max_theta 60
done