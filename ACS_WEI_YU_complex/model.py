import torch
import torch.nn as nn
from pytorch_lightning.core import LightningModule
# import scipy.linalg as sci
import numpy as np


class model(LightningModule):
    def __init__(self, **kwargs):
        super(model, self).__init__()
        self.save_hyperparameters()
        print('device: ', self.device)

        self.M = kwargs['M']
        self.P_dl = kwargs['P_dl']
        self.L = kwargs['beta_tr']
        self.K = kwargs['K']
        self.B = kwargs['B']
        self.snr_dl = kwargs['snr_dl']
        self.anneal_test = kwargs['annealing_rate_test']
        self.anneal_train =  kwargs['annealing_rate_train']
        self.anneal_rate = kwargs['annealing_rate']
        self.lr = kwargs['lr']
        self.b1 = kwargs['b1']
        self.b2 = kwargs['b2']
        self.T = kwargs['T']
        self.val_size = kwargs['val_size']
        self.test_size = kwargs['test_size']
        self.h_num = kwargs['h_num']

        ## pilot matrix
        DFT_Matrix = np.fft.fft(np.eye(self.M))
        X_init = DFT_Matrix[0::int(np.ceil(self.M / self.L)), :]
        if X_init.shape[0] != self.L:
            print('pilot error!') # (L(pilot_dim) * M )
        Xp_init = np.sqrt(self.P_dl / self.M) * X_init
        Xp_r_init = np.float32(np.real(Xp_init))
        Xp_i_init = np.float32(np.imag(Xp_init))

        Xp_r = nn.Parameter(torch.from_numpy(Xp_r_init), requires_grad=True)
        Xp_i = nn.Parameter(torch.from_numpy(Xp_i_init), requires_grad=True)

        #self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_parameter("Xp_r", Xp_r)
        self.register_parameter("Xp_i", Xp_i)


        ## UE network
        self.UE_network = self.construct_network_user()

        ## BS network
        self.BS_network_pre, self.BS_network_vr, self.BS_network_vi = self.construct_BS_network()


    def construct_network_user(self):
        network = nn.Sequential(
            nn.BatchNorm1d(self.L*2),

            nn.Linear(self.L*2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, self.B))
        return network

    def construct_BS_network(self):
        network_1 = nn.Sequential(
            nn.Linear(self.B* self.K, 1024*2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024*2),

            nn.Linear(1024*2, 512*2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512*2),

            nn.Linear(512*2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        network_v_r = nn.Sequential(
            nn.Linear(512, self.M * self.K))

        network_v_i = nn.Sequential(
            nn.Linear(512, self.M * self.K)
        )
        return network_1, network_v_r, network_v_i

        ## downlink training
    def DL_training_phase(self, hI, hR):
        y_nless = {0: 0}
        y_noisy = {0: 0}
        norm_X = torch.sqrt(torch.sum(torch.pow(torch.abs(self.Xp_r), 2) + torch.pow(torch.abs(self.Xp_i), 2), dim=1, keepdim=True))
        ## pilot matrix normalization
        self.Xp_r.data = np.sqrt(self.P_dl) * torch.divide(self.Xp_r, norm_X)
        self.Xp_i.data = np.sqrt(self.P_dl) * torch.divide(self.Xp_i, norm_X)
        power_X = torch.sum(torch.pow(torch.abs(self.Xp_r), 2) + torch.pow(torch.abs(self.Xp_i), 2), dim=1, keepdim=True)  # just to check!
        if torch.any(power_X < self.P_dl - 0.2) or torch.any(power_X > self.P_dl + 0.2):
            print('Pilot matrix power error!')

        X_p = self.Xp_r + 1j * self.Xp_i
        for kk in range(self.K):
            hR_temp = torch.reshape(hR[:, :, kk], [-1, self.M])
            hI_temp = torch.reshape(hI[:, :, kk], [-1, self.M])
            h = hR_temp + 1j * hI_temp
            y_tr = h @ torch.transpose(torch.conj(X_p), 1, 0) # h X^{\herm}
            y_nless[kk] = torch.cat((torch.real(y_tr), torch.imag(y_tr)), dim=1)
            y_noisy[kk] = y_nless[kk] + torch.normal(mean=torch.zeros(h.shape[0], 2 * self.L), std=0.5*torch.ones(h.shape[0], 2 * self.L)).to(self.device)
        return y_noisy, y_nless

    def UE_operations(self, y_noisy, anneal_rate):
        InfoBits = {0:0}
        for kk in range(self.K):
            InfoBits_linear = self.UE_network(y_noisy[kk])
            InfoBits_tanh = 2 * torch.sigmoid(anneal_rate * InfoBits_linear) -1
            InfoBits_sign = torch.sign(InfoBits_linear)
            InfoBits[kk] = InfoBits_tanh + (InfoBits_sign - InfoBits_tanh).detach()
            if kk == 0:
                DNN_input_BS = InfoBits[kk]
            else:
                DNN_input_BS = torch.cat((DNN_input_BS, InfoBits[kk]), dim=1)
        return DNN_input_BS


    def Rate_func_cal(self, hr, hi, V, noise_power, K, M):
        h = hr + 1j * hi
        h_hermitian = h.conj().transpose(2, 1)
        HH_V = torch.bmm(h_hermitian, V)
        norm2_hv = torch.pow(torch.real(HH_V), 2) + torch.pow(torch.imag(HH_V), 2)
        norm2_hv = torch.squeeze(norm2_hv)

        nom = norm2_hv[:, np.arange(K), np.arange(K)]
        denom = torch.sum(norm2_hv, dim=-1) - nom + noise_power  # batch_size * K
        rate = torch.log(1 + torch.divide(nom, denom)) / np.log(2.0)
        return rate

    def BS_operation(self, DNN_input_BS, hR, hI, train):
        output = self.BS_network_pre(DNN_input_BS)
        V_r = self.BS_network_vr(output) # dimension: batch_size * (M *K)
        V_i = self.BS_network_vi(output)

        V = V_r + 1j * V_i
        V = V.reshape(-1, self.M, self.K)

        norm_V = torch.sqrt(torch.sum(torch.norm(V, dim=1, keepdim=True) ** 2, dim=(1, 2), keepdim=True))
        V = torch.divide(V, norm_V) * np.sqrt(self.P_dl)

        rate = self.Rate_func_cal(hR, hI, V, 1, self.K, self.M)
        rate_mean = torch.mean(torch.sum(rate, dim=-1))
        return -rate_mean


    def forward(self, hR, hI, annealing_rate, train):
        y_noisy, y_nless = self.DL_training_phase(hI, hR)
        DNN_input_BS = self.UE_operations(y_noisy, annealing_rate)
        rate = self.BS_operation(DNN_input_BS, hR, hI, train)
        return rate

    def training_step(self, batch, batch_idx):
        hR, hI = batch
        hR = hR.float()
        hI = hI.float()
        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        position = np.random.permutation(hR.shape[0])
        rate = self(hR[position, :, :], hI[position, :, :], self.anneal_train, train=True)
        self.log('train_loss', rate, on_step=False, on_epoch=True)
        return rate

    def validation_step(self, batch, batch_idx):
        hR, hI = batch
        hR = hR.float()
        hI = hI.float()
        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        rate = self(hR, hI, self.anneal_test, train=False) # minus rate --> loss
        self.log('val_loss', rate, on_step=False, on_epoch=True, prog_bar=True)
        return rate*hR.shape[0]

    def validation_epoch_end(self, validation_steps_outputs):
        val_rate_stack = torch.stack(validation_steps_outputs)
        mean_val_rate = torch.sum(val_rate_stack)/(self.val_size*self.h_num)
        if (self.current_epoch)==0:
            self.best_loss = mean_val_rate
        else:
            if mean_val_rate >= self.best_loss:
                self.anneal_train = self.anneal_train * self.anneal_rate
            else:
                self.best_loss = mean_val_rate
        self.log('best_val_loss', self.best_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        hR, hI = batch
        hR = hR.float()
        hI = hI.float()
        hR = hR.view(-1, self.M, self.K)
        hI = hI.view(-1, self.M, self.K)
        rate = self(hR, hI, self.anneal_test, train=False)  # minus rate --> loss
        self.log('test_loss', -rate, on_step=False, on_epoch=True, prog_bar=True)
        return rate * hR.shape[0]

    def test_epoch_end(self, validation_steps_outputs):
        val_rate_stack = torch.stack(validation_steps_outputs)
        mean_val_rate = torch.sum(val_rate_stack) / (self.test_size * 10)
        self.test_rate = -mean_val_rate
        self.log('test_rate', -mean_val_rate, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return (
            {'optimizer': opt}
        )















