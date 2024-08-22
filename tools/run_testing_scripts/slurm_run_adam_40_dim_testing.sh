#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH -o TrainingLogs/test_40_dim_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/test_40_dim_error-%j          # send stderr to errfile
#SBATCH -J t40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com


# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

python tools/test.py /home/amakacs1/mmdetection/work_dirs/hyper_adam_40_dim_cosine_reduce_9/hyperbolic_config_adam_40_dim.py /home/amakacs1/mmdetection/work_dirs/hyper_adam_40_dim_cosine_reduce_9/epoch_12.pth --out /home/amakacs1/mmdetection/work_dirs/hyper_adam_40_dim_cosine_reduce_9/test/out.pickle --show-dir /home/amakacs1/mmdetection/work_dirs/hyper_adam_40_dim_cosine_reduce_9/test/vis/
