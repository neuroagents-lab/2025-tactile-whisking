#!/bin/bash

#SBATCH --job-name=tactile_tnn_%j        # Dynamic job name
#SBATCH --output=tactile_tnn_%j.out      # Output file (%j is the job ID)
#SBATCH --error=tactile_tnn_%j.err       # Error file
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=32                 # Allocate 32 CPU cores
#SBATCH --gres=gpu:A6000:1                 # Request 1 GPUs
#SBATCH --mem=48G                          # Request 48GB of memory
#SBATCH --time=48:00:00                    # Set a 48-hour time limit
#SBATCH --partition=general               # Specify the partition
#SBATCH --exclude=babel-6-17,babel-0-37,babel-1-31                     # Exclude specific node(s)

# Default values for parameters
n_times=22
input_shape='[30, 5, 7]'
run_name=null
model_config_file='model_configs/xxx.json'
data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed'
lr=0.1
optimizer='sgd'
scheduler='step_lr'

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

python main.py model.encoder.n_times="$n_times" \
  data.n_times="$n_times" \
  data.input_shape="$input_shape" \
  data.data_dir="$data_dir" \
  train.run_name="$run_name" \
  model.encoder.model_config_file="$model_config_file" \
  model.encoder.full_unroll=False \
  data.num_classes=117 \
  train.lr="$lr" \
  train.optimizer="$optimizer" \
  train.scheduler="$scheduler" \
  train.weight_decay=0.0001 \
  train.momentum=0.9 \
  train.step_size=30 \
  train.num_devices=1


# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.01:[this LARGELY improves over the baseline]
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='constant_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512_lr1e_2_constant' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'constant_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='constant_lr'-lr=0.005:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512_lr5e_3_constant' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.005 -o 'sgd' -s 'constant_lr'

# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_h512_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'


# UGRNN
# (sgd+step-lr)
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'

# (adam+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adam' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adam_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adam_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adam_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adam' -s 'step_lr'

# (adamw+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adamw' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'

# (adamw+step-lr+ln)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_ln_adamw' -m 'model_configs/rnn_fdb_converted_modern_ugrnn_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_ln_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_ugrnn_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_ln_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_ugrnn_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_ugrnn_ln_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_ugrnn_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'




# IntersectionRNN

# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'

# (adam+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adam' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adam_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adam_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adam_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adam' -s 'step_lr'

# (adamw+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'

# (adamw+step-lr+ln)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_ln' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_ln_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_ln_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_inter_adamw_ln_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'



# GRU

# (sgd+step-lr)
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'

# (adam+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adam' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adam_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adam_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adam_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adam' -s 'step_lr'

# (adamw+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adamw' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'

# (adamw+step-lr+ln)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_ln_adamw' -m 'model_configs/rnn_fdb_converted_modern_gru_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_ln_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_gru_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_ln_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_gru_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_gru_ln_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_gru_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'



# LSTM

# (sgd+step-lr)
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'

# (adam+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adam' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adam_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adam_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adam' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adam'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adam_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adam' -s 'step_lr'

# (adamw+step-lr)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adamw' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'

# (adamw+step-lr+ln)
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_ln_adamw' -m 'model_configs/rnn_fdb_converted_modern_lstm_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_ln_adamw_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_lstm_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_ln_adamw_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_lstm_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'adamw' -s 'step_lr'
# 1000hz, Zhuang+w. fdb---opt='adamw'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_shapenet.sh -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_zhuang_w_fdb_unroll22_lstm_ln_adamw_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_lstm_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'adamw' -s 'step_lr'


