import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import savemat
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
# from __init__ import *



if __name__ == '__main__':
    #####################################################main ############################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ##########################################cuda setting ############################################
    pl.seed_everything(6)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    B_list = np.array([1, 4, 8, 16, 25, 40])
    #B_list = np.array([1])
    folder_lists = sorted(os.listdir('./'))
    folder_required_lists = []
    for folder in folder_lists:
        if 'ACS' in folder:
            folder_required_lists.append(folder)

    #folder_required_lists = [folder_required_lists[2]]
    print(folder_required_lists)  # check labels
    rate_list = np.zeros((len(folder_required_lists) +2, len(B_list)))
    ####################################################################
    Lp = 2
    lr = 0.0001
    val_size = 1000
    test_size = 100
    M = 64
    K = 6
    beta_tr = 8
    snr_dl = 20
    batch_size = 1024
    n_epochs = 2000
    h_num = 1
    theta_max = 60

    for B_idx in range(len(B_list)):
        B = int(B_list[B_idx])
        # print(opt)
        #######################model#################
        for folder_idx in range(len(folder_required_lists)):
            folder = folder_required_lists[folder_idx]
            if 'WEI_YU' in folder:
                from ACS_WEI_YU_complex.model import *
                from ACS_WEI_YU_complex.__init__ import *

                opt = arg_generate()
                ##########################some parameter which requires computation ##########################
                # settings
                opt.train_size = 2
                opt.lr = lr
                opt.batch_size = batch_size
                opt.n_epochs = 5000
                opt.val_size = val_size
                opt.test_size = test_size
                opt.snr_dl = snr_dl
                opt.beta_tr = beta_tr
                opt.Lp_min = Lp
                opt.Lp_max = Lp
                opt.K = K
                opt.M = M
                opt.theta_max = theta_max

                opt.sample_num = 5
                opt.B = B
                opt.P_dl = 10 ** (opt.snr_dl / 10)
                opt.beta_fb = opt.beta_tr
                opt.h_num = h_num
                print(opt)

                val_data = data_generation(test=True, **vars(opt))
                print('len val: ', len(val_data))

                if torch.cuda.is_available():
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=16, shuffle=False,
                                            pin_memory=True)
                else:
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=8, shuffle=False,
                                            pin_memory=True)

                PATH = './' + folder + '/train_size_{}_M_{}_K_{}_beta_tr_{}_Lp_max_{}_Lp_min_{}_snr_dl_{}_beta_fb_{}_B_{}_h_num_{}_beta_0/batch_size_{}_lr_{}_epochs_{}/version_0/checkpoints/'.format(
                    opt.train_size, opt.M, opt.K,
                    opt.beta_tr, opt.Lp_max,
                    opt.Lp_min, opt.snr_dl,
                    opt.beta_fb, opt.B, opt.h_num,
                    opt.batch_size, opt.lr, opt.n_epochs)

                dir_list = os.listdir(PATH)
                extended_PATH = PATH + dir_list[0]
                CSI_feedback_model = model.load_from_checkpoint(checkpoint_path=extended_PATH, **vars(opt))
                print('model: ', CSI_feedback_model)

            elif 'only' in folder:
                from ACS_only_training_pilot_true.model import *
                from ACS_only_training_pilot_true.__init__ import *

                opt = arg_generate()
                ##########################some parameter which requires computation ##########################
                # settings
                opt.train_size = 20480
                opt.lr = lr
                opt.batch_size = batch_size
                opt.n_epochs = 500
                opt.val_size = val_size
                opt.test_size = test_size
                opt.snr_dl = snr_dl
                opt.beta_tr = beta_tr
                opt.Lp_min = Lp
                opt.Lp_max = Lp
                opt.K = K
                opt.M = M
                opt.theta_max = theta_max

                opt.sample_num = 5
                opt.B = B
                opt.P_dl = 10 ** (opt.snr_dl / 10)
                opt.beta_fb = opt.beta_tr
                opt.h_num = h_num
                print(opt)

                val_data = data_generation(test=True, **vars(opt))
                print('len val: ', len(val_data))

                if torch.cuda.is_available():
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=16, shuffle=False,
                                            pin_memory=True)
                else:
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=8, shuffle=False,
                                            pin_memory=True)

                PATH = './' + folder + '/train_size_{}_M_{}_K_{}_beta_tr_{}_Lp_max_{}_Lp_min_{}_snr_dl_{}_beta_fb_{}_B_{}_h_num_{}_beta_0_sample_num_{}/batch_size_{}_lr_{}_epochs_{}/version_0/checkpoints/'.format(
                    opt.train_size, opt.M, opt.K,
                    opt.beta_tr, opt.Lp_max,
                    opt.Lp_min, opt.snr_dl,
                    opt.beta_fb, opt.B, opt.h_num, opt.sample_num,
                    opt.batch_size, opt.lr, opt.n_epochs)
                dir_list = os.listdir(PATH)
                extended_PATH = PATH + dir_list[0]
                CSI_feedback_model = model.load_from_checkpoint(checkpoint_path=extended_PATH, **vars(opt))
                print('model: ', CSI_feedback_model)


            elif 'AF' in folder:
                from ACS_AF_lambda_all_1.__init__ import  *
                from ACS_AF_lambda_all_1.model import *

                opt = arg_generate()
                ##########################some parameter which requires computation ##########################
                # settings
                opt.train_size = 20480
                opt.lr = lr
                opt.batch_size = batch_size
                opt.n_epochs = n_epochs
                opt.val_size = val_size
                opt.test_size = test_size
                opt.snr_dl = snr_dl
                opt.beta_tr = beta_tr
                opt.Lp_min = Lp
                opt.Lp_max = Lp
                opt.K = K
                opt.M = M
                opt.theta_max = theta_max

                opt.sample_num = 5
                opt.B = B
                opt.P_dl = 10 ** (opt.snr_dl / 10)
                opt.beta_fb = opt.beta_tr
                opt.h_num = h_num
                print(opt)

                val_data = data_generation(test=True, **vars(opt))
                print('len val: ', len(val_data))

                if torch.cuda.is_available():
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=16, shuffle=False,
                                            pin_memory=True)
                else:
                    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=8, shuffle=False,
                                            pin_memory=True)

                CSI_feedback_model = model(**vars(opt))
                print('model: ', CSI_feedback_model)

            #####################################data ############################################

            if torch.cuda.is_available():
                trainer = Trainer(accelerator="gpu", devices=1, check_val_every_n_epoch=10,
                                  max_epochs=opt.n_epochs, log_every_n_steps=20,
                                  callbacks=[LearningRateMonitor("epoch"),
                                             ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
            else:
                trainer = Trainer(check_val_every_n_epoch=2, max_epochs=opt.n_epochs, log_every_n_steps=20,
                                  callbacks=[LearningRateMonitor("epoch"),
                                             ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])

            # ------------------------
            # 3 START TRAINING
            # ------------------------
            trainer.test(CSI_feedback_model, val_loader)

            rate_list[folder_idx, B_idx] = CSI_feedback_model.test_rate.detach().cpu().numpy()
            end_time = time.time()
                # print('time %.5f h ' % ((end_time - start_time) / 3600))


    # MRT _ ZF
    from MRT_ZF_results.MRT_ZF import *
    rate_MRT, rate_ZF = MRT_ZF(opt)
    print('rate_ZF: ', rate_ZF)

    #
    rate_list[-2, :] = rate_MRT * np.ones_like(B_list)
    rate_list[-1, :] = rate_ZF * np.ones_like(B_list)
    label_list = ['fixed pilot matrix without lambda design with MMSE and ZF', 'WEI YU', 'DLMMSE + ZF', 'MRT', 'ZF']
    #
    data = {'rate': rate_list, 'label': label_list}
    save_folder = './old_data_test_results_viz/'
    os.makedirs(save_folder, exist_ok=True)
    filename = save_folder + 'results_Lp_{}.mat'.format(Lp)
    savemat(filename, data)

    mpl.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes((0.1, 0.1, 0.5, 0.8))
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=3)
    plt.rc('patch', linewidth=3)
    plt.rc('legend', fancybox=True)

    color = ['-ro', '-g*', '-bv', '-y^', '--r*', '--go', '--bo', '--yv']
    for i in range(len(folder_required_lists)):
        plt.plot(B_list, rate_list[i, :], color[i], label=label_list[i])

    plt.plot(B_list, rate_MRT*np.ones(len(B_list)), color[-2],  label='MRT with perfect DL CSI')
    plt.plot(B_list, rate_ZF * np.ones(len(B_list)), color[-1], label='ZF with perfect DL CSI')

    plt.xlabel('B (bits)')
    plt.ylabel('DL Sum Rates (bits)')
    plt.legend(loc='best')
    filename = 'M {} K {} Lp min {} Lp max {} beta tr {} snr dl {}'.format(M, K, Lp, Lp, beta_tr, snr_dl)
    plt.title(filename)
    plt.grid(True)
    plt.savefig(save_folder+ 'results_Lp_{}.pdf'.format(Lp), bbox_inches='tight')














