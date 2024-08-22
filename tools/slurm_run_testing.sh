#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH -o TrainingLogs/test_0_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/test_0_error-%j          # send stderr to errfile
#SBATCH -J test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com

pwd

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

# python tools/test.py test_folder/config.py work_dirs/config/epoch_6.pth

python tools/test.py test_folder/config.py test_folder/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --out ./work_dirs/coco_detection/out.pickle

# python tools/test.py test_folder/config.py test_folder/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --show --show-dir ./results/
# python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --show --show-dir results/
