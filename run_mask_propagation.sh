#!/bin/bash
#SBATCH --job-name=mask_prop
#SBATCH --account=3152128
#SBATCH --partition=gpu,gpunew
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=../slogs/%x_%j.out
#SBATCH --error=../slogs/%x_%j.err
#SBATCH --mail-user mark.lomele@gmail.com
#SBATCH --mail-type END

python examples/run_egoexo_propagation.py \
    --data_dir /data/video_datasets/3321908/output_dir_all \
    --input_csv mask_prop_input.csv \
    --out_root /data/video_datasets/reluminati/propagation_results \
    --vis_stride 10 \
    --confidence_threshold 0.3 \
    --out_root /mask_prop_exp