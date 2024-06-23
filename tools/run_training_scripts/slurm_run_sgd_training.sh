#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -o TrainingLogs/hyperbolic_faster_rcnn_hyper_sgd_001_cosine_reduce_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/hyperbolic_faster_rcnn_hyper_sgd_001_cosine_reduce_error-%j          # send stderr to errfile
#SBATCH -J sgd_001_cosine_reduce
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com


# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/hyperbolic_faster_rcnn/sgd/hyperbolic_config_sgd.py --work-dir /home/amakacs1/mmdetection/work_dirs/hyper_sgd_001_cosine_reduce/