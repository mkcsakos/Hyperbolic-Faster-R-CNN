#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -o TrainingLogs/hyperbolic_faster_rcnn_v12_cls_fc_100x100_warmup_sgd_005_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/hyperbolic_faster_rcnn_v12_cls_fc_100x100_warmup_sgd_005_error-%j          # send stderr to errfile
#SBATCH -J v12_cls_fc_1024x100_warmup_sgd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com


pwd

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3


python -u tools/train.py

# python -u tools/train.py test_folder/config.py --work-dir ./work_dirs/config/hyper_test/


# python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/faster-rcnn_r50_fpn_1x_coco/


# python -u tools/train.py /home/amakacs1/mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_c4-1x_coco.py --work-dir ./work_dirs/faster-rcnn_r50-caffe_c4-1x_coco/
# python -u tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/config/
