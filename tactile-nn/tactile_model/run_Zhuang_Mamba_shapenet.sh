#!/bin/bash

#SBATCH --job-name=tactile_enc_Zhuang_att_Mamba_%j        # Dynamic job name
#SBATCH --output=tactile_enc_Zhuang_att_Mamba_%j.out      # Output file (%j is the job ID)
#SBATCH --error=tactile_enc_Zhuang_att_Mamba_%j.err       # Error file
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=32                 # Allocate 32 CPU cores
#SBATCH --gres=gpu:A6000:1                 # Request 1 GPUs
#SBATCH --mem=48G                          # Request 48GB of memory
#SBATCH --time=48:00:00                    # Set a 48-hour time limit
#SBATCH --partition=general               # Specify the partition
#SBATCH --exclude=babel-6-17,babel-0-35,babel-1-31,shire-1-1,babel-0-37,babel-15-36  # Exclude specific node(s)

# Default values for parameters
n_times=22
input_shape='[30, 5, 7]'
run_name=null
model_config_file='model_configs/xxx.json'
data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed'
lr=0.0001
optimizer='adamw'
scheduler='cos_w_warmup'

# Parse flags
while getopts n:i:r:m:d:l:o:s: flag
do
    case "${flag}" in
        n) n_times=${OPTARG};;
        i) input_shape=${OPTARG};;
        r) run_name=${OPTARG};;
        m) model_config_file=${OPTARG};;
        d) data_dir=${OPTARG};;
        l) lr=${OPTARG};;
        o) optimizer=${OPTARG};;
        s) scheduler=${OPTARG};;
        *)
            echo "Invalid option: -$flag" >&2
            exit 1;;
    esac
done

# Print out the parameters
echo "Running with the following parameters:"
echo "  n_times: $n_times"
echo "  input_shape: $input_shape"
echo "  run_name: $run_name"
echo "  model_config_file: $model_config_file"
echo "  data_dir: $data_dir"
echo "  lr: $lr"
echo "  optimizer: $optimizer"
echo "  scheduler: $scheduler"

# Run the Python script with specified arguments

python main.py model=enc_Zhuang_att_mamba \
  data.input_shape="$input_shape" \
  data.n_times="$n_times" \
  data.data_dir="$data_dir" \
  train.run_name="$run_name" \
  model.encoder.model_config_file="$model_config_file" \
  data.num_classes=117 \
  train.lr="$lr" \
  train.optimizer="$optimizer" \
  train.scheduler="$scheduler" \
  train.weight_decay=0.0001 \
  train.momentum=0.9 \
  train.step_size=30 \
  train.num_devices=1

# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='cos_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'cos_w_warmup'

# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_steplr_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_steplr_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_steplr_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'

# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_sgd_steplr_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_sgd_steplr_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_sgd_steplr_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'


### with causal-conv1d installed
# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='cos_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'cos_w_warmup'

# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_steplr_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_steplr_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_steplr_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'

# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_sgd_steplr_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_sgd_steplr_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, enc_Zhuang_att_Mamba---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_Mamba_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_Mamba_causal_sgd_steplr_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

