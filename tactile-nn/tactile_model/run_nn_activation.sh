#!/bin/bash

#SBATCH --job-name=tactile-nn-%j
#SBATCH --output=tactile_nn_%j.out
#SBATCH --error=tactile_nn_%j.err 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32 
#SBATCH --gres=gpu:1 
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --mail-type=END
#SBATCH --mail-user=trinityc@cmu.edu


# conda init
source /home/trinityc/miniconda3/etc/profile.d/conda.sh
conda activate /home/trinityc/miniconda3/envs/tactile-nn

cd /home/trinityc/tactile/tactile-nn/tactile_model

### supervised learning (best)

# Zhuang's model
python get_nn_activation.py model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_h512_lr1e_2'

# Zhuang's model with UGRNN
python get_nn_activation.py model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adamw_lr1e_3'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_ln_lr1e_3'

# Zhuang's model with GRU
python get_nn_activation.py model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_gru_adamw_lr1e_3'

# Zhuang's model with LSTM
python get_nn_activation.py model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_lstm_adamw_lr1e_3'


# BAKU
python get_nn_activation.py model=baku data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku'

# Zhuang+Mamba
python get_nn_activation.py model=enc_Zhuang_att_mamba model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_sgd_steplr_lr1e_3'

# Zhuang+GPT
python get_nn_activation.py model=enc_Zhuang_att_GPT model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT'

# Zhuang's model with UGRNN+GPT
python get_nn_activation.py model=enc_Zhuang_att_GPT model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_ugrnn_att_GPT'

# Zhuang's model with IntersectionRNN[+LN]+GPT
python get_nn_activation.py model=enc_Zhuang_att_GPT model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_inter_ln_att_GPT'

# Zhuang's model with GRU+GPT
python get_nn_activation.py model=enc_Zhuang_att_GPT model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_gru_att_GPT'

# Zhuang's model with LSTM+GPT
python get_nn_activation.py model=enc_Zhuang_att_GPT model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_lstm_att_GPT'


# ResNet+Mamba
python get_nn_activation.py model=enc_resnet18_att_mamba data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_sgd_steplr_lr1e_3'

# ResNet
python get_nn_activation.py model=vanilla_resnet18 data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_pt_resnet18_sgd_steplr_lr1e_1'


### SimCLR (best)

# Zhuang's model
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_simclr_rot_tflip110_lr1e_1'

# Zhuang's model with UGRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_ugrnn_simclr_rot_tflip110_lr1e_2'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_inter_ln_simclr_rot_tflip110_lr1e_1'

# Zhuang's model with GRU
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_gru_simclr_rot_tflip110_lr1e_1'

# Zhuang's model with LSTM
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_lstm_simclr_rot_tflip110_lr1e_2'

# BAKU
python get_nn_activation.py train.ssl_finetune=True model=baku model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku_simclr_rot_tflip110_lr1e_1'

# Zhuang+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_mamba model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_simclr_rot_tflip110_lr1e_2'

# Zhuang+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT_simclr_rot_tflip110_lr1e_2'

# Zhuang's model with Intersection+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_4'

# ResNet+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_resnet18_att_mamba model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_simclr_rot_tflip110_lr1e_2'



### SimSiam (best)

# Zhuang's model
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with UGRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_ugrnn_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_inter_ln_simsiam_lbs_rot_tflip110_lr1e_3'

# Zhuang's model with GRU
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_gru_simsiam_lbs_rot_tflip110_lr1e_2'

# Zhuang's model with LSTM
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_lstm_simsiam_lbs_rot_tflip110_lr1e_1'

# BAKU
python get_nn_activation.py train.ssl_finetune=True model=baku model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku_simsiam_lbs_rot_tflip110_lr1e_2'

# Zhuang+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_mamba model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_simsiam_lbs_rot_tflip110_lr1e_2'

# Zhuang+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT_simsiam_lbs_rot_tflip110_lr1e_3'

# Zhuang's model with Intersection+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simsiam_lbs_rot_tflip110_lr1e_3'

# ResNet+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_resnet18_att_mamba model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_simsiam_lbs_rot_tflip110_lr1e_2'



### AutoEncoding (best)

# Zhuang's model
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_AE_rot_tflip110_lr1e_1'

# Zhuang's model with UGRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_ugrnn_AE_rot_tflip110_lr1e_1'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_inter_ln_AE_rot_tflip110_lr1e_4'

# Zhuang's model with GRU
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_gru_AE_rot_tflip110_lr1e_1'

# Zhuang's model with LSTM
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_lstm_AE_rot_tflip110_lr1e_1'

# BAKU
python get_nn_activation.py train.ssl_finetune=True model=baku model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku_AE_rot_tflip110_lr1e_1'

# Zhuang+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_mamba model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_AE_rot_tflip110_lr1e_4'

# Zhuang+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT_AE_rot_tflip110_lr1e_2'

# ResNet+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_resnet18_att_mamba model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_AE_rot_tflip110_lr1e_2'



### SimSiam with warm-up epochs=10 (best)

# Zhuang's model
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with UGRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_ugrnn_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_inter_ln_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with GRU
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_gru_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'

# Zhuang's model with LSTM
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_Zhuang_lstm_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'


# BAKU
python get_nn_activation.py train.ssl_finetune=True model=baku model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku_w_wu_simsiam_lbs_rot_tflip110_lr1e_1'


# Zhuang+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_mamba model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_w_wu_simsiam_lbs_rot_tflip110_lr1e_2'


# Zhuang+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_2'


# ResNet+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_resnet18_att_mamba model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_w_wu_simsiam_lbs_rot_tflip110_lr1e_2'



### AutoEncoding (best) w. SSL-transform

# Zhuang's model
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_w_fdb_unroll22_AE_rot_tflip110_lr1e_1'

# Zhuang's model with UGRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_ugrnn.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_ugrnn_AE_w_ssl_rot_tflip110_lr1e_1'

# Zhuang's model with IntersectionRNN
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_inter_ln_AE_w_ssl_rot_tflip110_lr1e_4'

# Zhuang's model with GRU
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_gru.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_gru_AE_w_ssl_rot_tflip110_lr1e_1'

# Zhuang's model with LSTM
python get_nn_activation.py train.ssl_finetune=True model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_lstm.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_zhuang_lstm_AE_w_ssl_rot_tflip110_lr1e_1'


# BAKU
python get_nn_activation.py train.ssl_finetune=True model=baku model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_baku_AE_w_ssl_rot_tflip110_lr1e_4'

# Zhuang+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_mamba model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_Mamba_AE_w_ssl_rot_tflip110_lr1e_4'

# Zhuang+GPT
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_att_GPT_AE_w_ssl_rot_tflip110_lr1e_1'

# ResNet+Mamba
python get_nn_activation.py train.ssl_finetune=True model=enc_resnet18_att_mamba model/decoder=linear data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_pt_resnet18_att_Mamba_AE_w_ssl_rot_tflip110_lr1e_2'

# Zhuang's model with Intersection+GPT+SimCLR
python get_nn_activation.py train.ssl_finetune=True model=enc_Zhuang_att_GPT model/decoder=linear model.encoder.n_times=22 model.encoder.model_config_file='model_configs/rnn_fdb_converted_modern_intersection_ln.json' data.input_shape='[30, 5, 7]' data.num_classes=117 data.data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' train.run_name='tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_4'
