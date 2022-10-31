# from __init__ import *
import torch
import numpy as np
import os

def Rate_func_cal(hr, hi, V, noise_power, K, M):
    h = hr + 1j * hi
    h_hermitian = h.conj().transpose(2, 1)
    HH_V = torch.bmm(h_hermitian, V)
    norm2_hv = torch.pow(torch.real(HH_V), 2) + torch.pow(torch.imag(HH_V), 2)
    norm2_hv = torch.squeeze(norm2_hv)

    nom = norm2_hv[:, np.arange(K), np.arange(K)]
    denom = torch.sum(norm2_hv, dim=-1) - nom + noise_power  # batch_size * K
    rate = torch.log(1 + torch.divide(nom, denom)) / np.log(2.0)
    return rate

# opt = arg_generate()
#
# opt.P_dl = 10 ** (opt.snr_dl / 10)
# snr_ul_power = opt.kappa * opt.P_dl
# opt.snr_ul = 10 * np.log10(snr_ul_power)

def MRT_ZF(opt):
    # data load
    data_filename = './test_data_M_{}_K_{}_Lp_min_{}_Lp_max_{}_h_num_{}_sample_num_{}.npz'.format(
                    opt.M, opt.K, opt.Lp_min, opt.Lp_max, 10, opt.sample_num)

    data = np.load(data_filename)
    h_act_test = data['h']
    hR_act_test = np.real(h_act_test)
    hI_act_test = np.imag(h_act_test)


    hR_act_test = hR_act_test.reshape(-1, opt.M, opt.K)
    hI_act_test = hI_act_test.reshape(-1, opt.M, opt.K)
    hR = torch.from_numpy(hR_act_test)
    hI = torch.from_numpy(hI_act_test)

    opt.P_dl = 10 ** (opt.snr_dl / 10)
    ################################# MRT-baseline
    V_MRT = hR + 1j * hI  # batch_size * M * K V_MRT = H^ {\herm}
    # norm_V = torch.sqrt(torch.sum(torch.pow(torch.real(V_MRT), 2) + torch.pow(torch.imag(V_MRT), 2), dim=1, keepdim=True))
    # V_MRT = np.sqrt(opt.P_dl / opt.K) * torch.divide(V_MRT, norm_V)

    norm_V = torch.sqrt(torch.sum(torch.norm(V_MRT, dim=1, keepdim=True) ** 2, dim=(1, 2), keepdim=True))
    V_MRT = torch.divide(V_MRT, norm_V) * np.sqrt(opt.P_dl)

    rate_UP = Rate_func_cal(hR, hI, V_MRT, 1, opt.K, opt.M)
    rate_UP_mean = torch.mean(torch.sum(rate_UP, dim=-1))
    #####################################ZF ################
    ch_size = hR.shape[0]
    V_ZF_r = torch.zeros_like(hR)  # batch_size * M * K
    V_ZF_i = torch.zeros_like(hI)
    print('ch sample: ', ch_size)
    for ch in range(ch_size):
        H = hR[ch, :, :] + 1j * hI[ch, :, :]
        HH = H.conj().transpose(1, 0)
        V_ZF = torch.linalg.pinv(HH)  # M * K

        norm_V_ZF = torch.sqrt(torch.sum(torch.norm(V_ZF, dim=0, keepdim=True) ** 2, dim=(0, 1), keepdim=True))
        V_ZF = torch.divide(V_ZF, norm_V_ZF) * np.sqrt(opt.P_dl)

        V_ZF_r[ch, :, :] = torch.real(V_ZF)  # batch_size * M * K
        V_ZF_i[ch, :, :] = torch.imag(V_ZF)

    V_ZF_all = V_ZF_r + 1j * V_ZF_i
    rate_ZF = Rate_func_cal(hR, hI, V_ZF_all, 1, opt.K, opt.M)
    rate_ZF_mean = torch.mean(torch.sum(rate_ZF, dim=-1))

    print('Lp: ', opt.Lp_min)
    print('ZF rate: ', rate_ZF_mean)
    print('MRT rate: ', rate_UP_mean)

    return rate_UP_mean, rate_ZF_mean
#
# from __init__ import  *
# opt = arg_generate()
# rate, rate_ZF = MRT_ZF(opt)
