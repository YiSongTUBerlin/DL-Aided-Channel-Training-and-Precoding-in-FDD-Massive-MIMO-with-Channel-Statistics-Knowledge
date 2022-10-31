import numpy as np
import argparse
import time
import os
from numpy import linalg
from scipy.linalg import dft
import torch
from scipy.linalg import toeplitz
from numpy import linalg as LA
import matplotlib.pyplot as plt

#
# def Toep(X):
#     M = X.shape[0]
#     loop_time = 200
#     for i in range(loop_time):
#         x = np.array([np.mean(np.diag(X, -j)) for j in range(M)])
#         x[0] = np.real(x[0])
#         X = toeplitz(x)
#         w, V = np.linalg.eig(X)
#         if min(w) >=0:
#             break
#         else:
#             w[w<0] = 0
#             X = V @ np.diag(w) @ V.conj().T
#     return X

def Toep(X):
    M = X.shape[0]
    x = np.array([np.mean(np.diag(X, -j)) for j in range(M)])
    x[0] = np.real(x[0])
    X = toeplitz(x)
    return X

def generate_batch_data(size, M,K,L_min, L_max, LSF_UE, Mainlobe_UE, HalfBW_UE, theta_max, N_ul, h_num=0):
    F_dl = np.complex64(np.fft.fft(np.eye(M)) / np.sqrt(M))
    h_act = np.complex64(np.zeros((size, h_num, M, K)))  # F^{\herm} @ h
    x_act = np.complex64(np.zeros((size, h_num, M, K)))  # F^{\herm} @ h
    x_cov_from_ul_toep = np.complex64(np.zeros((size, h_num, M, M, K)))
    x_cov_from_dl_Sigma = np.complex64(np.zeros((size, h_num, M, M, K)))
    true_dl_Sigma = np.complex64(np.zeros((size, h_num, M, M, K)))
    ul_toep_Sigma = np.complex64(np.zeros((size, h_num, M, M,  K)))
    ul_dl_ratio = 0.9
    from0toM = np.float32(np.arange(0, M, 1))
    for size_idx in range(size):
        for kk in range(K):
            L = np.random.randint(L_max - L_min + 1) + L_min
            alpha_act = (np.random.randn(h_num, L) + 1j * np.random.randn(h_num, L)) / np.sqrt(2)
            theta_act = (np.pi / 180) * np.random.uniform(low=Mainlobe_UE[kk] - HalfBW_UE[kk],
                                                          high=Mainlobe_UE[kk] + HalfBW_UE[kk], size=[L, 1])
            gamma_input = np.random.uniform(0.5, 0.8, L)
            gamma_input = gamma_input / np.sum(gamma_input)
            diag_gamma = np.diag(np.sqrt(gamma_input))
            theta_act_expanded_temp = np.tile(theta_act, (1, M))
            #### UL and DL samples
            response_temp_DL = np.exp(-1j * np.pi * np.sin(theta_act_expanded_temp)/np.sin(theta_max/180*np.pi) * from0toM)  ### dimension: L * M
            response_temp_UL = np.exp(-1j * np.pi * ul_dl_ratio * np.sin(theta_act_expanded_temp)/np.sin(theta_max/180*np.pi) * from0toM)  ### dimension: L * M
            ####################################################current DL CSI ##################################
            h_dl = alpha_act @ diag_gamma @ response_temp_DL  # h_num * M
            h_act[size_idx, :, :, kk] = h_dl
            x_act[size_idx, :, :, kk] = h_dl @ F_dl.conj()

            #
            # color = ['-b*', '-ro', '-g^', '-yv', '--b*', '--ro', '--g^', '--yv',]
            # plt.figure()
            # for i in range(8):
            #     plt.plot(np.arange(M), np.abs(x_act[size_idx, i, :, kk]), color[i])
            # plt.show()

            # UL samples to DL covariance matrix
            alpha_ul_act = (np.random.randn(h_num + N_ul, L) + 1j * np.random.randn(h_num + N_ul, L)) / np.sqrt(2)
            h_ul_samples = alpha_ul_act @ diag_gamma @ response_temp_UL  # samples * M
            sample_cov_UL = np.array([1 / N_ul * (h_ul_samples[i:i + N_ul, :]).T @ (h_ul_samples[i:i + N_ul, :]).conj() for i in range(h_num)])
            toep_psd_UL = np.array([Toep(sample_cov_UL[i]) for i in range(h_num)])

            ul_toep_Sigma[size_idx, :, :, :, kk] = toep_psd_UL
            x_cov_from_ul_toep[size_idx, :, :, :, kk] = np.array([F_dl.conj().T @ toep_psd_UL[i]@ F_dl for i in range(h_num)])

            Sigma_dl = np.transpose(response_temp_DL) @ np.diag(gamma_input) @ np.conjugate(response_temp_DL)
            true_dl_Sigma[size_idx, :, :, :, kk] = np.tile(Sigma_dl, (h_num, 1, 1))
            x_cov_from_dl_Sigma[size_idx, :, :, :, kk] = np.tile(F_dl.conj().T @ Sigma_dl@ F_dl, (h_num, 1, 1))

    return h_act, x_act, true_dl_Sigma, ul_toep_Sigma, x_cov_from_dl_Sigma, x_cov_from_ul_toep


if __name__ == '__main__':
    # freeze_support()
    start_time = time.time()
    #####################################################main ############################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    'Learning Parameters'
    parser.add_argument("--test_size", type=int, default=10, help="number of training set and validation set")
    parser.add_argument("--val_size", type=int, default=10, help="number of training set and validation set")

    'System Parameters'
    parser.add_argument("--M", type=int, default=64, help="Antenna Number")
    parser.add_argument("--K", type=int, default=6, help="number of users")
    parser.add_argument("--Lp_max", type=int, default=2, help="max number of path")
    parser.add_argument("--Lp_min", type=int, default=2, help="min number of path")
    parser.add_argument("--h_num", type=int, default=10, help="the number of collected previous UL channel samples")
    parser.add_argument("--sample_num", type=int, default=5, help="the number of collected previous UL channel samples")

    parser.add_argument("--LSF_UE", type=np.array, default=np.array([0.0,0.0],dtype=np.float32), help="Mean of path gains for K users")
    parser.add_argument("--Mainlobe_UE", type=np.array, default=np.array([0,0],dtype=np.float32), help="Center of the AoD range for K users")
    parser.add_argument("--HalfBW_UE", type=np.array, default= np.array([30.0,30.0],dtype=np.float32), help="Half of the AoD range for K users")
    parser.add_argument("--max_theta", type=float, default=60, help="Mean of path gains for K users")

    parser.add_argument("--data_file_num", type=int, default=2, help="the number of collected previous UL channel samples")

    opt = parser.parse_args()
    opt.LSF_UE = np.zeros(opt.K)
    opt.Mainlobe_UE = np.zeros(opt.K)
    opt.HalfBW_UE = opt.max_theta * np.ones(opt.K)

    np.random.seed(6)


    h_act, x_act, true_dl_Sigma, ul_toep_Sigma, x_cov_from_dl_Sigma, x_cov_from_ul_toep = generate_batch_data(opt.val_size,opt.M, opt.K, opt.Lp_min, opt.Lp_max,opt.LSF_UE, opt.Mainlobe_UE, opt.HalfBW_UE, opt.max_theta,  opt.sample_num, opt.h_num)

    filename = './val_data_M_{}_K_{}_Lp_min_{}_Lp_max_{}_h_num_{}_sample_num_{}.npz'.format(
        opt.M, opt.K, opt.Lp_min, opt.Lp_max, opt.h_num, opt.sample_num)
    np.savez(filename,  h=h_act, Sigma_dl=true_dl_Sigma, Sigma_ul_toep=ul_toep_Sigma, x=x_act, x_cov_from_ul_toep=x_cov_from_ul_toep,x_cov_from_dl_Sigma=x_cov_from_dl_Sigma)

    h_act, x_act, true_dl_Sigma, ul_toep_Sigma, x_cov_from_dl_Sigma, x_cov_from_ul_toep = generate_batch_data(opt.test_size,opt.M, opt.K, opt.Lp_min, opt.Lp_max,opt.LSF_UE, opt.Mainlobe_UE, opt.HalfBW_UE,  opt.max_theta, opt.sample_num, opt.h_num)

    filename = './test_data_M_{}_K_{}_Lp_min_{}_Lp_max_{}_h_num_{}_sample_num_{}.npz'.format(
        opt.M, opt.K, opt.Lp_min, opt.Lp_max, opt.h_num, opt.sample_num)
    np.savez(filename,  h=h_act, Sigma_dl=true_dl_Sigma, Sigma_ul_toep=ul_toep_Sigma, x=x_act, x_cov_from_ul_toep=x_cov_from_ul_toep,x_cov_from_dl_Sigma=x_cov_from_dl_Sigma)

    #train_data
    # for i in range(opt.data_file_num):
    #     print('i: ', i)
    #     h_act_train, x_train, gaussian_vector_dl_N_from_UL_train = generate_batch_data(opt.train_size,opt.M, opt.K, opt.Lp_min, opt.Lp_max,opt.LSF_UE, opt.Mainlobe_UE, opt.HalfBW_UE, opt.sample_list, opt.h_num)
    #
    #     filename = '../train_data_M_{}_K_{}_Lp_min_{}_Lp_max_{}_h_num_{}_sample_list_{}_idx_{}.npz'.format(
    #         opt.M, opt.K, opt.Lp_min, opt.Lp_max, opt.h_num, opt.sample_list, i)
    #     np.savez(filename, h_act=h_act_train,  x=x_train, gaussian_vec=gaussian_vector_dl_N_from_UL_train)
    #
