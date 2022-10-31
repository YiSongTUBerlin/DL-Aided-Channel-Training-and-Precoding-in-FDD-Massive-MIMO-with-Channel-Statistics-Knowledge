import argparse
import numpy as np
from torch.utils.data import Dataset

def arg_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--batch_num_per_epoch", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--train_size", type=int, default=2, help="number of training set and validation set")
    parser.add_argument("--sample_per_file", type=int, default=5000, help="number of training set and validation set")
    parser.add_argument("--val_size", type=int, default=5, help="number of training set and validation set")
    parser.add_argument("--test_size", type=int, default=500, help="number of training set and validation set")
    parser.add_argument("--start_epoch", type=int, default=100, help="number of training set and validation set")
    parser.add_argument("--sample_num", type=int, default=5, help="number of training set and validation set")
    parser.add_argument("--LSF_UE", type=np.array, default=np.array([0.0,0.0],dtype=np.float32), help="Mean of path gains for K users")
    parser.add_argument("--Mainlobe_UE", type=np.array, default=np.array([0,0],dtype=np.float32), help="Center of the AoD range for K users")
    parser.add_argument("--HalfBW_UE", type=np.array, default= np.array([30.0,30.0],dtype=np.float32), help="Half of the AoD range for K users")


    parser.add_argument("--annealing_rate", type=float, default=1.001, help="Annealing Rate")
    parser.add_argument("--annealing_rate_test", type=float, default=1, help="Annealing Rate param in testing")
    parser.add_argument("--annealing_rate_train", type=float, default=1, help="Annealing Rate Param in training")

    'System Parameter'
    parser.add_argument("--M", type=int, default=64, help="Antenna Number")
    parser.add_argument("--P_dl", type=int, default=1, help="POWER")
    parser.add_argument("--beta_tr", type=int, default=8, help="number of pilots")  # 'this is beta_tr'
    parser.add_argument("--K", type=int, default=6, help="number of users")
    parser.add_argument("--B", type=int, default=40, help="feedback capacity bits")
    parser.add_argument("--beta", type=int, default=0, help="the lagrangian multipler of objective function")
    parser.add_argument("--Lp_max", type=int, default=2, help="max number of path")
    parser.add_argument("--Lp_min", type=int, default=2, help="min number of path")
    parser.add_argument("--h_num", type=int, default=10, help="the number of collected previous UL channel samples")
    # parser.add_argument("--channel_sample", type=int, default=20, help="the instaneous samples of same geometry")
    parser.add_argument("--beta_fb", type=int, default=8, help="the feedback dimension") # can be tuned
    parser.add_argument("--snr_dl", type=int, default=10, help="SNR in dB")
    parser.add_argument("--kappa", type=float, default=1, help="ratio snr_ul /snr_dl not in dB")
    parser.add_argument("--T", type=int, default=70, help="the total number of dimension") # 14 * 5

    opt = parser.parse_args()
    # opt.sample_list = [5, 100]
    opt.LSF_UE = np.zeros(opt.K)
    opt.Mainlobe_UE = np.zeros(opt.K)
    opt.HalfBW_UE = 60 * np.ones(opt.K)
    return opt



