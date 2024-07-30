#!/bin/bash
#SBATCH --job-name=upper_verde_ensemble
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=1-2%2

# Calculate the task ID based on the SLURM_ARRAY_TASK_ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))

runname='upper_verde_2006'

# Generate a unique output file name based on the task ID
out_dir="/home/ab6361/hydrogen_workspace/subset_ensembles/ensemble_results/${runname}_${task_id}"

# Run your Python script with the selected input file and output file
python ./src/cli.py \
    --out_dir="$out_dir" \
    --runname=${runname} \
    --start="2005-10-01" \
    --end="2006-09-30" \
    --huc_id='15060202' \
    --number_runs=2 \
    --modify_indicator \
    --modify_parameters
