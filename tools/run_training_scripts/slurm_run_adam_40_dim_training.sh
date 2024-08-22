#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -o TrainingLogs/hyperbolic_faster_rcnn_hyper_adam_40_dim_cosine_reduce_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/hyperbolic_faster_rcnn_hyper_adam_40_dim_cosine_reduce_error-%j          # send stderr to errfile
#SBATCH -J hyper_adam_40_dim_cosine_reduce
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com


# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3


# python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/hyperbolic_faster_rcnn/tmp_configs/hyperbolic_config_adamw.py --work-dir /home/amakacs1/mmdetection/work_dirs/hyper_adamW_00005_cosine_reduce_1/
# python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/hyperbolic_faster_rcnn/tmp_configs/hyperbolic_config_sgd.py --work-dir /home/amakacs1/mmdetection/work_dirs/hyper_sgd_001_cosine_reduce_1/
python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/hyperbolic_faster_rcnn/adam_40_dim/hyperbolic_config_adam_40_dim.py --work-dir /home/amakacs1/mmdetection/work_dirs/hyper_adam_40_dim_cosine_reduce/
