import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from __init__ import *
import os


if __name__ == '__main__':
    # freeze_support()
    start_time = time.time()
    #####################################################main ############################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = arg_generate()
    torch.autograd.set_detect_anomaly(True)

    ##########################some parameter which requires computation ##########################
    opt.P_dl = 10**(opt.snr_dl/10)
    opt.beta_fb = opt.beta_tr

    ##########################################cuda setting ############################################
    pl.seed_everything(6)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ########################data ##################################
    # hR_all, hI_all, sample_cov_data, sigma_dl_data = generate_all_data(**vars(opt))
    val_data = data_generation(test=False, train=False, **vars(opt))
    opt.val_size = len(val_data)
    print('len test: ', len(val_data))

    if torch.cuda.is_available():
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=16, shuffle=False, pin_memory=True)
    else:
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=4, shuffle=False)

    print(opt)
    #######################model#################
    CSI_feedback_model = model(**vars(opt))
     ###############model initialization
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0)

    CSI_feedback_model.apply(initialize_weights)
    print('model: ', CSI_feedback_model)


    if torch.cuda.is_available():
        trainer = Trainer(accelerator="gpu", devices=1, check_val_every_n_epoch=10,
                          max_epochs=opt.n_epochs,  log_every_n_steps=20,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=20,  mode="min"),
                                     LearningRateMonitor("epoch"),
                                     ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
    else:
        trainer = Trainer( check_val_every_n_epoch=5, max_epochs=opt.n_epochs, log_every_n_steps=20, detect_anomaly=True,
                          callbacks=[LearningRateMonitor("epoch"),
                                     ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.test(CSI_feedback_model, val_loader)
    rate = CSI_feedback_model.test_rate
    print('rate: ', rate)
    end_time = time.time()
    print('time %.5f h ' %((end_time-start_time)/3600))












