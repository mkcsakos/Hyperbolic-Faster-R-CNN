#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH -o TrainingLogs/slurm_run_training_0_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/slurm_run_training_0_error-%j          # send stderr to errfile
#SBATCH -J run_training_
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com


pwd

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

echo "running faster-RCNN training"

python -u tools/run_training_.py
# python -u tools/train.py test_folder/config.py --work-dir ./work_dirs/config/hyper_test/
# python -u tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/config/
