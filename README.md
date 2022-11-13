# Deep-Learning-Aided-Channel-Training-and-Precoding-in-FDD-Massive-MIMO-with-Channel-Statistics-Knowledge

This is the code for the paper titled "Deep-Learning Aided Rate Optimization in FDD Massive MIMO with Channel Statistics Knowledge", which can be found on the following link：

The code has 6 parts:

  1. generate_test_data.py: to generate the validation data and test data where Lp =2 or Lp= 20, h_num= 10 (the number of channel sample per geometry), M = 64, and K =6. 

  2. ACS_only_training_pilot_true: this folder provides the code for our proposed method, where the main file is FDD_mmwave_precoding_torch.py.
  
  3. ACS_WEI_YU_complex: this folder provides the code for retraining the model propsed from paper "Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO".
  
    ** job_test_B.sh: this file is aimed at training our proposed method and WEI YU method under different feedback capacity B. 
  
  4. ACS_AF_Lambda_all_1: this folder provides the code for the analog feedback scheme which assumes that lambda are all ones, uses MMSE estimation for channel estimation, and applies Zero-forcing precoding on the estimate channels. 
  
  5. MRT_ZF_results: This folder provides the code for generating the sum-rate for MRT precoding under perfect CSI and ZF precoding under perfect CSI. 
  
  6. load_model_test.py: this python file can reproduce the simulation results in Fig. 2 and Fig. 3 where B = {1, 4, 8, 16, 25, 40}. 
 
  
  
If you have further questions about the codes, please write an email to: yi.song@tu-berlin.de
