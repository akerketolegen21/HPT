#!/bin/bash
#SBATCH --job-name=hpt_train_captions      # Job name
#SBATCH --output=hpt_train_captions_%j.out      # Standard out`put log file (%j is job ID)
#SBATCH --error=hpt_train_captions_%j.err       # Standard error log file (%j is job ID)
#SBATCH --partition=testing        # Partition to submit the job to
#SBATCH --ntasks=1                   # Number of tasks (1 for a single task)
#SBATCH --cpus-per-task=4          # Number of CPUs per task (adjust as needed)
#SBATCH --mem=40G                    # Memory per node
#SBATCH --time=2-23:00:00              # Maximum run time (adjust as needed)


# module load python/3.10.16
# module load pytorch/2.6.0
# module load cuda/12.4
# module load torchvision/0.13.1
# module load transformers/4.25.1
# module load datasets/2.11.0
# module load torch_scatter/2.0.9
# module load torch_geometric/2.0.4
# module load torch_sparse/0.6.18
# module load wandb
# source ~/.conda/envs/torch_env/bin/activate

# echo "Start to preprocess data"
# python convert_data.py \
#     --train datas/train_pngpt_capt.json \
#     --val datas/val_ngptcapt.json \
#     --output_dir data/proc_pngptc/ \
#     --dataset_name pngptc


echo "Start to train model"
python train.py \
    --data data/proc_pngptc/ \
    --batch 16 \
    --name hpt-pngptc \
    --resume 

# echo "Make predictions"
# python predict.py
# echo "FIN"