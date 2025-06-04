#!/bin/bash

#SBATCH --job-name=tactile_tnn_ssl_%j        # Dynamic job name
#SBATCH --output=tactile_tnn_ssl_%j.out      # Output file (%j is the job ID)
#SBATCH --error=tactile_tnn_ssl_%j.err       # Error file
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

## Run the Python script with specified arguments
#echo  " running ssl training"
#python main.py model=enc_Zhuang_att_GPT model/decoder=empty_ssl model.encoder.n_times="$n_times" \
#  data.n_times="$n_times" \
#  data.input_shape="$input_shape" \
#  data.data_dir="$data_dir" \
#  train.run_name="$run_name" \
#  train.ssl="$ssl" \
#  train.ssl_temp_tran="$use_temp_tran" \
#  train.ssl_finetune=False \
#  model.encoder.model_config_file="$model_config_file" \
#  model.encoder.full_unroll=False \
#  data.num_classes=117 \
#  train.lr="$lr" \
#  train.batch_size="$batch_size" \
#  train.optimizer="$optimizer" \
#  train.scheduler="$scheduler" \
#  train.weight_decay=0.0001 \
#  train.momentum=0.9 \
#  train.step_size=30 \
#  train.warmup_epochs=10 \
#  train.warmup_ratio=0.0001 \
#  train.min_lr=0.0 \
#  train.num_devices=4

echo  " running linear fine-tuning"
python main.py model=enc_Zhuang_att_GPT model/decoder=linear train.ssl_finetune=True train.ssl=False \
  model.encoder.n_times="$n_times" \
  data.n_times="$n_times" \
  data.input_shape="$input_shape" \
  data.data_dir="$data_dir" \
  train.run_name="$run_name" \
  model.encoder.model_config_file="$model_config_file" \
  model.encoder.full_unroll=False \
  data.num_classes=117 \
  train.lr=0.1 \
  train.batch_size=256 \
  train.optimizer="sgd" \
  train.scheduler="step_lr" \
  train.weight_decay=0.0001 \
  train.momentum=0.9 \
  train.step_size=30 \
  train.num_devices=1


## SimSiam w. bs=1024 (w. rot+110tflip)  # The batch size below is correct, as we manually set the num_devices=2 when training with SimSiam
# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simsiam_lbs_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simsiam_lbs_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simsiam_lbs_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simsiam_lbs_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'cos_annealing_w_warmup'


## SimCLR w. bs=256 (w. rot+110tflip)
# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'lars' -s 'cos_annealing_w_warmup'


### With warm-up
## SimSiam w. bs=1024 (w. rot+110tflip)  # The batch size below is correct, as we manually set the num_devices=2 when training with SimSiam
# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 512 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'cos_annealing_w_warmup'


## AutoEncoder w. bs=256 (w. rot+110tflip)
# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'


## random init + linear probe
# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'RandomLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_rdm_init' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang-UGRNN+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'RandomLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_ugrnn_att_GPT_rdm_init' -m 'model_configs/rnn_fdb_converted_modern_ugrnn.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang-Inter-LN+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'RandomLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_rdm_init' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang-GRU+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'RandomLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_gru_att_GPT_rdm_init' -m 'model_configs/rnn_fdb_converted_modern_gru.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang-LSTM+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'RandomLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_lstm_att_GPT_rdm_init' -m 'model_configs/rnn_fdb_converted_modern_lstm.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'



## SimCLR w. bs=256 (w. img transformation)
# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_img_trans_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_img_trans_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_img_trans_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_simclr_img_trans_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'lars' -s 'cos_annealing_w_warmup'


# AutoEncoding w. ssl transform
# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_w_ssl_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_w_ssl_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_w_ssl_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'step_lr'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='step_lr'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'AutoEncoderLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_att_GPT_AE_w_ssl_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'step_lr'



### Zhuang+Inter+LN+GPT
## SimCLR w. bs=256 (w. rot+110tflip)
# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'lars' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang_GPT---opt='lars'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimCLRLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simclr_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'lars' -s 'cos_annealing_w_warmup'


## SimSiam w. bs=1024 (w. rot+110tflip)
# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simsiam_lbs_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simsiam_lbs_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simsiam_lbs_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_simsiam_lbs_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'cos_annealing_w_warmup'

## with warm up
# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.1:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_1' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.1 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.01:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_2' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.01 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_3' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.001 -o 'sgd' -s 'cos_annealing_w_warmup'

# 1000hz, Zhuang+GPT---opt='sgd'-sch='cos_annealing_w_warmup'-lr=0.0001:
# sbatch run_Zhuang_GPT_ssl_shapenet.sh -L 'SimSiamLoss' -t True -b 256 -n 22 -i '[30, 5, 7]' -r 'tactile1000hz_enc_Zhuang_inter_ln_att_GPT_w_wu_simsiam_lbs_rot_tflip110_lr1e_4' -m 'model_configs/rnn_fdb_converted_modern_intersection_ln.json' -d '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed' -l 0.0001 -o 'sgd' -s 'cos_annealing_w_warmup'

