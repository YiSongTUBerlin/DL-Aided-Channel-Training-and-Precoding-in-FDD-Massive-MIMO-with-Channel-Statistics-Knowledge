import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# warnings.filterwarnings("ignore")
from pytorch_lightning.core import LightningModule
import numpy as np
import time
import seaborn as sns
import pandas as pd


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

        Psi_init = np.complex64((np.random.randn(self.M, self.L) + 1j* np.random.randn(self.M, self.L))/ np.sqrt(2*self.M))
        Psi = torch.from_numpy(Psi_init)
        self.register_buffer("Psi_mat", Psi)

        self.pilot_matrix_FNN = nn.Sequential(
            nn.Linear(self.M * self.K*2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, self.M),
            nn.Tanh()
        )


    ## downlink training
    def rate_optimization(self, hR, hI, x_cov, sigma, anneal_rate, plot=False):
        # start_time = time.time()
        sigma = sigma.view(-1, self.M * self.K)
        sigma_matrix = torch.cat((torch.real(sigma), torch.imag(sigma)), dim=-1)
        Lambda_output = self.pilot_matrix_FNN(sigma_matrix)  # FNN network
        Lambda_output = 1/2*(Lambda_output + 1)

        # Lambda_output = torch.ones_like(Lambda_output)
        # end_time = time.time()
        # print('lambda: ', (end_time - start_time) / 3600)
        #############################plot the heatmap of selection vector##########################################################
        if plot:
            save_data_file = './only_train_lambda/'
            os.makedirs(save_data_file, exist_ok=True)
            Lambda_list = Lambda_output.detach().cpu().numpy()
            fig = plt.figure()
            lambda_pd = pd.DataFrame(Lambda_list[np.arange(50)*10, :],columns=np.arange(self.M) + 1)
            lambda_pd.index = np.arange(50) + 1
            ax = sns.heatmap(lambda_pd, linewidth=0.5, xticklabels=5, yticklabels=4, vmin=0, vmax=1)
            plt.savefig(save_data_file + 'heatmap_B_{}_Lp_{}.pdf'.format(self.B, self.Lp_max), bbox_inches='tight')
            plt.close(fig)


        # start_time = time.time()
        diag_b_matrix = torch.zeros(sigma_matrix.shape[0], self.M, self.M, dtype=torch.complex64).to(self.device)
        diag_b_matrix[:, np.arange(self.M), np.arange(self.M)] = Lambda_output + 1j * torch.zeros_like(Lambda_output)
        Psi_mat_batch = self.Psi_mat.repeat(sigma_matrix.shape[0], 1, 1)
        X_p = torch.bmm(diag_b_matrix, Psi_mat_batch)
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
            feedback = torch.sqrt(P_ul) * raw_feedback + (torch.randn_like(raw_feedback) + 1j * torch.randn_like(raw_feedback)) * np.sqrt(1 / 2)

            # channel MMSE estimator based on UL received symbols
            sigma_matrix_k = x_cov[:, :, :, kk]
            sigma_matrix_X_p = torch.bmm(sigma_matrix_k, X_p)
            X_herm_sigma_X_p = torch.bmm(X_p_herm, sigma_matrix_X_p)
            sigma_fb_inv = torch.linalg.inv(X_herm_sigma_X_p + (1 + 1/P_ul)* torch.eye(self.beta_fb).repeat(hR.shape[0], 1, 1).to(self.device))
            A_mat = torch.bmm(sigma_matrix_X_p, sigma_fb_inv)/(torch.sqrt(P_ul))
            est_x = torch.bmm(A_mat, feedback)
            # effective channel
            est_x = torch.bmm(diag_b_matrix, est_x) # batch* M * 1
            # est_x = torch.squeeze(est_x)
            #
            # error = torch.norm(est_x - h_complex[:, :, kk], dim=-1)**2 / torch.norm(h_complex[:, :, kk], dim=-1)**2
            # print('error: ', error.mean())

            # feedback_info = torch.cat((torch.real(est_x), torch.imag(est_x)), dim=-1)

            if kk == 0:
                feedback_all_user= est_x
            else:
                feedback_all_user = torch.cat((feedback_all_user, est_x), dim=-1)

        # Bs produce precoder

        # output = self.BS_network(feedback_all_user)
        # output = output.view(-1, 2 * self.M, self.K)
        # V = output[:, :self.M, :] + 1j * output[:, self.M:, :]

        V = torch.linalg.pinv(torch.transpose(feedback_all_user.conj(), 2, 1))
        V_0 = torch.bmm(diag_b_matrix, V)  # batch * M * K
        norm_V_0 = torch.sqrt(torch.sum(torch.norm(V_0, dim=1, keepdim=True) ** 2, dim=(1, 2), keepdim=True))
        new_V = torch.divide(V_0, norm_V_0) * np.sqrt(self.P_dl)

        # end_time = time.time()
        # print('precoder: ', (end_time - start_time) / 3600)

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


    def forward(self, hR, hI, x_cov, sigma, train, test=False):
        if train:
            anneal_rate = self.anneal_train
        else:
            anneal_rate = self.anneal_test
        if test:
            plot= True
        else:
            plot = False
        rate = self.rate_optimization(hR, hI, x_cov, sigma, anneal_rate, plot)
        return rate

    def training_step(self, batch, batch_idx):
        hR, hI, x_cov, sigma = batch
        hR = hR.float()
        hI = hI.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        x_cov = x_cov.view(-1, self.M, self.M, self.K)
        sigma = sigma.view(-1, self.M,  self.K)

        rate = self(hR, hI, x_cov, sigma, train=True)
        self.log('train_loss', -rate, on_step=False, on_epoch=True, prog_bar=True)
        return -rate

    def validation_step(self, batch, batch_idx):
        hR, hI, x_cov, sigma = batch
        hR = hR.float()
        hI = hI.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        x_cov = x_cov.view(-1, self.M, self.M, self.K)
        sigma = sigma.view(-1, self.M,  self.K)

        rate = self(hR, hI, x_cov, sigma, train=True)
        self.log('val_loss', -rate, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        hR, hI, x_cov, sigma = batch
        hR = hR.float()
        hI = hI.float()

        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        x_cov = x_cov.view(-1, self.M, self.M, self.K)
        sigma = sigma.view(-1, self.M,  self.K)

        rate = self(hR, hI, x_cov, sigma, train=False, test=True)
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















