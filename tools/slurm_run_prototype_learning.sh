#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH -o TrainingLogs/test_0_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/test_0_error-%j          # send stderr to errfile
#SBATCH -J train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=makacsakos@gmail.com

pwd

# conda init bash
source /home/${USER}/.bashrc
conda activate mmcv_3


python /home/amakacs1/mmdetection/mmdet/utils/prototype_learning.py -d 20 -c 81 -r /home/amakacs1/mmdetection/hyperbolic_assets/prototypes/


# sys.argv.append('-c')
    # sys.argv.append('81')
    # sys.argv.append("-d")
    # sys.argv.append('2')
    # sys.argv.append("-r")
    # sys.argv.append("/home/amakacs1/mmdetection/hyperbolic_assets/prototypes/")

# python -u tools/train.py test_folder/config.py --work-dir ./work_dirs/config/hyper_test/
# python -u tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/config/
