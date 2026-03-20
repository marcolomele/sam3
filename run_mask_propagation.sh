#!/bin/bash
#SBATCH --job-name=mask_prop
#SBATCH --account=3152128
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=../slogs/%j.out
#SBATCH --error=../slogs/%j.err
#SBATCH --mail-user mark.lomele@gmail.com
#SBATCH --mail-type END

module load cuda/12.4

eval "$(conda shell.bash hook)"
conda activate sam3_py11

python examples/run_egoexo_propagation.py \
    --data_dir ../data/video_datasets/3321908/output_dir_all \
    --input_csv examples/mask_prop_input.csv \
    --out_root propagation_results \
    --vis_stride 10 \
    --confidence_threshold 0.3 \
    --bpe_path /home/3152128/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
