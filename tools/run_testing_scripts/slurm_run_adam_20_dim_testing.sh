#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH -o TestingLogs/test_20_dim_out-%j          # send stdout to outfile
#SBATCH -e TestingLogs/test_20_dim_error-%j          # send stderr to errfile
#SBATCH -J t20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

python tools/test.py /home/amakacs1/mmdetection/work_dirs/hyper_adam_20_dim_cosine_reduce_1/hyperbolic_config_adam_20_dim.py /home/amakacs1/mmdetection/work_dirs/hyper_adam_20_dim_cosine_reduce_1/epoch_12.pth --out /home/amakacs1/mmdetection/work_dirs/hyper_adam_20_dim_cosine_reduce_1/test/out.pickle --show-dir /home/amakacs1/mmdetection/work_dirs/hyper_adam_20_dim_cosine_reduce_1/test/vis/
