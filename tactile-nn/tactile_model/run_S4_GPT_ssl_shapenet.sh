#!/bin/bash

#SBATCH --job-name=tactile_tnn_ssl_%j        # Dynamic job name
#SBATCH --output=tactile_tnn_ssl_%j.out      # Output file (%j is the job ID)
#SBATCH --error=tactile_tnn_ssl_%j.err       # Error file
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=32                 # Allocate 32 CPU cores
#SBATCH --gres=gpu:A6000:2                 # Request 1 GPUs
#SBATCH --mem=96G                          # Request 48GB of memory
#SBATCH --time=48:00:00                    # Set a 48-hour time limit
#SBATCH --partition=general               # Specify the partition
#SBATCH --exclude=babel-6-17,babel-0-35,babel-1-31,shire-1-1,babel-0-37,babel-15-36  # Exclude specific node(s)

# Default values for parameters
n_times=22
input_shape='[30, 5, 7]'
run_name=null
model_config_file='model_configs/xxx.json'
data_dir='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed'
lr=0.1
optimizer='sgd'
scheduler='step_lr'
ssl='SimCLRLoss'
use_temp_tran=True
batch_size=256

# Parse flags
while getopts n:i:r:m:d:l:o:s:L:t:b: flag
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
        L) ssl=${OPTARG};;
        t) use_temp_tran=${OPTARG};;
        b) batch_size=${OPTARG};;
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
echo "  ssl: $ssl"
echo "  use_temp_tran: $use_temp_tran"
echo "  batch_size: $batch_size"

# Run the Python script with specified arguments
echo  " running ssl training"
python main.py model=enc_S4_att_GPT model/decoder=empty_ssl \
  data.n_times="$n_times" \
  data.input_shape="$input_shape" \
  data.data_dir="$data_dir" \
  train.run_name="$run_name" \
  train.ssl="$ssl" \
  train.ssl_temp_tran="$use_temp_tran" \
  train.ssl_finetune=False \
  data.num_classes=117 \
  train.lr="$lr" \
  train.batch_size="$batch_size" \
  train.optimizer="$optimizer" \
  train.scheduler="$scheduler" \
  train.weight_decay=0.0001 \
  train.momentum=0.9 \
  train.step_size=30 \
  train.warmup_epochs=10 \
  train.warmup_ratio=0.0001 \
  train.min_lr=0.0 \
  train.num_devices=2

echo  " running linear fine-tuning"
python main.py model=enc_S4_att_GPT model/decoder=linear train.ssl_finetune=True train.ssl=False \
  data.n_times="$n_times" \
  data.input_shape="$input_shape" \
  data.data_dir="$data_dir" \
  train.run_name="$run_name" \
  data.num_classes=117 \
  train.lr=0.1 \
  train.batch_size=256 \
  train.optimizer="sgd" \
  train.scheduler="step_lr" \
  train.weight_decay=0.0001 \
  train.momentum=0.9 \
  train.step_size=30 \
  train.num_devices=1


## SimSiam w. bs=1024 (w. rot+110tflip)
# 1000hz, S4+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simsiam_lbs_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, S4+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simsiam_lbs_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, S4+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simsiam_lbs_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, S4+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simsiam_lbs_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'cos_annealing_w_warmup'


## SimCLR w. bs=256 (w. rot+110tflip)
# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simclr_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_Mamba---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simclr_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_Mamba---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simclr_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_Mamba---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_S4_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_S4_att_GPT_simclr_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'lars' -s 'cos_annealing_w_warmup'



