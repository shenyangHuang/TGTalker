#/bin/bash
#SBATCH --partition=main  #unkillable #main #long
#SBATCH --output=wiki_qwen38b_.txt 
#SBATCH --error=wiki_qwen38b_error.txt   
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=32G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00                       # The job will run for 1 day

module load python/3.10
source $SCRATCH/vllm_env/bin/activate

pwd
CUDA_VISIBLE_DEVICES=0 python -u reasoning_main_feat.py --batch 200 --model qwen8b --in_size 5 --bg_size 300 --data tgbl-wiki --nbr 2 --icl --logfile wiki_qwen38b_log.json
