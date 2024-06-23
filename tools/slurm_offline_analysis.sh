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
#SBATCH -—mail-type=ALL
#SBATCH -—mail-user=makacsakos@gmail.com

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3

# python tools/analysis_tools/analyze_results.py \
#     test_folder/config.py \
#     work_dirs/coco_detection/out.pickle \
#     work_dirs/coco_detection/results_1 \
#     --topk 50

# python tools/analysis_tools/analyze_results.py \
#     test_folder/config.py \
#     work_dirs/coco_detection/out.pickle \
#     work_dirs/coco_detection/results_2 \
#     --show-score-thr 0.3

# python tools/analysis_tools/confusion_matrix.py \
#     test_folder/config.py \
#     work_dirs/coco_detection/out.pickle \
#     work_dirs/coco_detection/results_3 \
#     --show

python tools/test.py \
    test_folder/config.py \
    test_folder/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --eval bbox
