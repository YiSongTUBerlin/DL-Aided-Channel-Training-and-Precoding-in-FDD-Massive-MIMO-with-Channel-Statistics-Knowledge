import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# warnings.filterwarnings("ignore")
from pytorch_lightning.core import LightningModule
import numpy as np
import time
import seaborn as sns


class model(LightningModule):
    def __init__(self, **kwargs):
        super(model, self).__init__()
        self.save_hyperparameters()
        print('device: ', self.device)

        self.M = kwargs['M']
        self.P_dl = kwargs['P_dl']
        self.L = kwargs['beta_tr']
        self.K = kwargs['K']
        self.snr_dl = kwargs['snr_dl']
        self.lr = kwargs['lr']
        self.b1 = kwargs['b1']
        self.b2 = kwargs['b2']
        self.beta = kwargs['beta']
        self.T = kwargs['T']
        self.sample_num = kwargs['sample_num']
        self.Lp_min = kwargs['Lp_min']
        self.Lp_max = kwargs['Lp_max']

        self.anneal_test = kwargs['annealing_rate_test']
        self.anneal_train = kwargs['annealing_rate_train']
        self.anneal_rate = kwargs['annealing_rate']

        self.beta_fb = kwargs['beta_fb']
        self.B = kwargs['B']

        P_ul = (2 ** (self.B / self.beta_fb) - 1)
        print('UL SNR: ', 10 * np.log(1 + P_ul)/np.log(10))

        self.val_size = kwargs['val_size']
        self.test_size = kwargs['test_size']
        self.h_num = kwargs['h_num']

        U_init = np.fft.fft(np.eye(self.M)) / np.sqrt(self.M)
        U_init = np.array(U_init, dtype=np.complex64)
        U = torch.from_numpy(U_init)
        self.register_buffer("UR", torch.real(U))
        self.register_buffer("UI", torch.imag(U))
        self.register_buffer("U_mat", self.UR + 1j* self.UI)


    ## downlink training
    def rate_optimization(self, hR, hI, gaussian_vec, anneal_rate, plot=False):
        lambda_vec = torch.ones(hR.shape[0], self.M, dtype=torch.complex64)
        # s = 64
        # if s != 64:
        #     zero_num  = self.M - s
        #     position = np.random.permutation(self.M)[:zero_num]
        #     lambda_vec[:, position] = torch.zeros(hR.shape[0], zero_num, dtype=torch.complex64)
        lambda_diag_matrix = torch.zeros(hR.shape[0], self.M, self.M, dtype=torch.complex64)
        lambda_diag_matrix[:, np.arange(self.M), np.arange(self.M)] = lambda_vec
        Psi_mat_batch = (torch.randn(hR.shape[0], self.M, self.L) + 1j * torch.randn(hR.shape[0], self.M, self.L)) / np.sqrt(2)

        #Psi_mat_batch = (torch.randn(hR.shape[0], self.M, self.L) +1j * torch.zeros(hR.shape[0], self.M, self.L))/ np.sqrt(self.M)

        F_mat = self.U_mat.repeat(hR.shape[0], 1, 1)
        F_diag_mat = torch.bmm(F_mat, lambda_diag_matrix)

        X_p = torch.bmm(F_diag_mat, Psi_mat_batch)
        norm_X_p = torch.norm(X_p, dim=1, keepdim=True)
        X_p = np.sqrt(self.P_dl) * torch.divide(X_p, norm_X_p)
        # end_time = time.time()
        # print('pilot_matrix: ', (end_time - start_time) / 3600)

        # start_time = time.time()
        X_p_herm = torch.transpose(X_p, 2, 1).conj()  # Phi B h = Psi \Lambda U^h h ---> B = Lambda U^h __> precoder h^{\herm} V -- > V = B^{\herm} V
        h_complex = hR + 1j * hI
        y_tr = torch.bmm(X_p_herm, h_complex)
        y_noisy = y_tr + (torch.randn_like(y_tr) + 1j * torch.randn_like(y_tr)) * np.sqrt(1/2)
        # end_time = time.time()
        # print('DL training: ', (end_time - start_time) / 3600)

        for kk in range(self.K):
            raw_feedback = y_noisy[:, :, kk]
            raw_feedback = torch.unsqueeze(raw_feedback, dim=2)
            power_feedback = torch.norm(raw_feedback, dim=(1, 2), keepdim=True) ** 2 / self.beta_fb
            P_ul = (2 ** (self.B / self.beta_fb) - 1) / power_feedback
            feedback = torch.sqrt(P_ul) * raw_feedback

            # channel MMSE estimator based on UL received symbols
            sigma_matrix_k = gaussian_vec[:, :, :, kk]
            sigma_matrix_X_p = torch.bmm(sigma_matrix_k, X_p)
            X_herm_sigma_X_p = torch.bmm(X_p_herm, sigma_matrix_X_p)
            sigma_fb_inv = torch.linalg.inv(X_herm_sigma_X_p + (1 + 1/P_ul)* torch.eye(self.beta_fb).repeat(hR.shape[0], 1, 1).to(self.device))
            A_mat = torch.bmm(sigma_matrix_X_p, sigma_fb_inv)/(torch.sqrt(P_ul))
            est_h = torch.bmm(A_mat, feedback + (torch.randn_like(feedback) + 1j *torch.randn_like(feedback)) * np.sqrt(1 / 2))  # batch * M * 1
            est_x = torch.bmm(torch.transpose(F_diag_mat.conj(), 2, 1), est_h)

            error = torch.mean(torch.linalg.norm(torch.squeeze(est_h) - h_complex[:, :, kk], dim=-1) **2/torch.linalg.norm(h_complex[:, :, kk], dim=-1)**2)
            print('CE error: ', error)


            if kk == 0:
                feedback_all_user= est_x
            else:
                feedback_all_user = torch.cat((feedback_all_user, est_x), dim=-1) # batch * M * K

        # Bs produce precoder
        H_herm = torch.transpose(feedback_all_user.conj(), 2, 1)
        V_0 = torch.linalg.pinv(H_herm)
        V_0 = torch.bmm(F_diag_mat, V_0)
        norm_V_0 = torch.sqrt(torch.sum(torch.norm(V_0, dim=1, keepdim=True) ** 2, dim=(1, 2), keepdim=True))
        new_V = torch.divide(V_0, norm_V_0) * np.sqrt(self.P_dl)

        rate = self.Rate_func(hR, hI, new_V, 1, self.K, self.M)
        rate_mean = torch.mean(torch.sum(rate, dim=-1))

        return rate_mean # batch_size * (2 * self.L) * self.K

    def Rate_func(self, hr, hi, V, noise_power, K, M):
        h = hr + 1j * hi
        h_hermitian = h.conj().transpose(2, 1)
        HH_V = torch.bmm(h_hermitian, V)
        norm2_hv = torch.pow(torch.real(HH_V), 2) + torch.pow(torch.imag(HH_V), 2)
        norm2_hv = torch.squeeze(norm2_hv)

        nom = norm2_hv[:, np.arange(K), np.arange(K)]
        denom = torch.sum(norm2_hv, dim=-1) - nom + noise_power  # batch_size * K
        rate = torch.log(1 + torch.divide(nom, denom)) / np.log(2.0)
        return rate


    def forward(self, hR, hI, gaussian_vec, train, test=False):
        if train:
            anneal_rate = self.anneal_train
        else:
            anneal_rate = self.anneal_test
        if test:
            plot= True
        else:
            plot = False
        rate = self.rate_optimization(hR, hI, gaussian_vec, anneal_rate, plot)
        return rate

    def training_step(self, batch, batch_idx):
        hR, hI, gaussian_vec = batch
        hR = hR.float()
        hI = hI.float()
        # gaussian_vec = gaussian_vec.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        gaussian_vec = gaussian_vec.view(-1, self.M, self.M, self.K)

        rate = self(hR, hI, gaussian_vec, train=True)
        self.log('train_loss', -rate, on_step=False, on_epoch=True, prog_bar=True)
        return -rate

    def validation_step(self, batch, batch_idx):
        hR, hI, gaussian_vec = batch
        hR = hR.float()
        hI = hI.float()
        # gaussian_vec = gaussian_vec.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        gaussian_vec = gaussian_vec.view(-1, self.M, self.M, self.K)

        rate = self(hR, hI, gaussian_vec, train=True)
        self.log('val_loss', -rate, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        hR, hI, gaussian_vec = batch
        hR = hR.float()
        hI = hI.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        gaussian_vec = gaussian_vec.view(-1, self.M, self.M, self.K)

        rate = self(hR, hI, gaussian_vec, train=True, test=True)
        self.log('test_rate', rate, on_step=False, on_epoch=True, prog_bar=True)
        return rate * hR.shape[0]

    def test_epoch_end(self, validation_steps_outputs):
        val_rate_stack = torch.stack(validation_steps_outputs)
        mean_val_rate = torch.sum(val_rate_stack) / (self.test_size * 10)
        self.test_rate = mean_val_rate
        self.log('mean_rate', mean_val_rate, on_step=False, on_epoch=True, prog_bar=True)

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            # print('name: ', name)
            self.logger.experiment.add_histogram(name + '_gradient', params.grad, self.current_epoch)

    def training_epoch_end(self, output):
        self.custom_histogram_adder()


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return (
            {'optimizer': opt}
        )















