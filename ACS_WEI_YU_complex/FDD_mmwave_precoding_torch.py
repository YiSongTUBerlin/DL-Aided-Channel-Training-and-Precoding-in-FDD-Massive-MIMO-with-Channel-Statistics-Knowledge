import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
from pytorch_lightning.trainer import Trainer
from __init__ import *
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import time


if __name__ == '__main__':
    #####################################################main ############################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = arg_generate()
    ##########################some parameter which requires computation ##########################
    opt.P_dl = 10**(opt.snr_dl/10)
    # snr_ul_power = opt.kappa * opt.P_dl
    # opt.snr_ul = 10 * np.log10(snr_ul_power)
    opt.beta_fb = opt.beta_tr

    ##########################################cuda setting ############################################
    pl.seed_everything(6)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ########################data ##################################
    train_data = data_generation(test=False, train=True, **vars(opt))
    print('len train: ', len(train_data))
    val_data = data_generation(test=False, train=False, **vars(opt))
    print('len val: ', len(val_data))


    if torch.cuda.is_available():
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=5, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=opt.batch_size,  num_workers=8, shuffle=True, pin_memory=True)

    if torch.cuda.is_available():
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=5, shuffle=False, pin_memory=True)
    else:
        val_loader = DataLoader(val_data, batch_size=opt.batch_size,  num_workers=8, shuffle=False, pin_memory=True)


    print(opt)
    start_time = time.time()
    #######################model#################
    CSI_feedback_model = model(**vars(opt))
     ###############model initialization

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            #nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0)


    CSI_feedback_model.apply(initialize_weights)
    print('model: ', CSI_feedback_model)

    # ----------------------------
    # 2 INIT TRAINER
    # ----------------------------
    name = 'batch_size_{}_lr_{}_epochs_{}'.format(opt.batch_size, opt.lr, opt.n_epochs)

    logger = TensorBoardLogger("./ACS_WEI_YU_complex/train_size_{}_M_{}_K_{}_beta_tr_{}_Lp_max_{}_Lp_min_{}_snr_dl_{}_beta_fb_{}_B_{}_h_num_{}_beta_{}".format(opt.train_size, opt.M, opt.K, opt.beta_tr, opt.Lp_max, opt.Lp_min, opt.snr_dl,
        opt.beta_fb, opt.B, opt.h_num, opt.beta), name=name)

    # EarlyStopping(monitor="val_loss", patience=10,  mode="min"),
    if torch.cuda.is_available():
        trainer = Trainer(accelerator="gpu", devices=1, logger=logger, check_val_every_n_epoch=10,
                          max_epochs=opt.n_epochs, log_every_n_steps=20,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=20,  mode="min"),
                                     LearningRateMonitor("epoch"),
                                     ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
    else:
        trainer = Trainer(logger=logger, check_val_every_n_epoch=2, max_epochs=opt.n_epochs, log_every_n_steps=20,
                          callbacks=[LearningRateMonitor("epoch"),
                                     ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(CSI_feedback_model, train_loader, val_loader)
    # trainer.test(CSI_feedback_model, test_loader)
    end_time = time.time()
    print('time %.5f h ' % ((end_time - start_time) / 3600))


    # trainer.test(val_loader)













