#!/bin/bash
#SBATCH --job-name=french_broad_ensemble
#SBATCH --ntasks=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=6-11%1

# Calculate the task ID based on the SLURM_ARRAY_TASK_ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))

runname='french_broad_2005'

# Generate a unique output file name based on the task ID
out_dir="/home/ab6361/hydrogen_workspace/subset_ensembles/ensemble_results/${runname}_${task_id}"

# Run your Python script with the selected input file and output file
python ./src/cli.py \
    --out_dir="$out_dir" \
    --runname=${runname} \
    --start="2004-10-01" \
    --end="2005-09-30" \
    --huc_id=060101 \
    --number_runs=2 \
    --modify_indicator \
    --modify_parameters
