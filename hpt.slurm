#!/bin/bash
#SBATCH --job-name=hpt_train      # Job name
#SBATCH --output=hpt_train_%j.out      # Standard output log file (%j is job ID)
#SBATCH --error=hpt_train_%j.err       # Standard error log file (%j is job ID)
#SBATCH --partition=testing        # Partition to submit the job to
#SBATCH --ntasks=1                   # Number of tasks (1 for a single task)
#SBATCH --cpus-per-task=4          # Number of CPUs per task (adjust as needed)
#SBATCH --mem=40G                    # Memory per node
#SBATCH --time=12:00:00              # Maximum run time (adjust as needed)


# module load python/3.10.16
# module load pytorch/1.12.1
# module load cuda/11.8
# module load torchvision/0.13.1
# module load transformers/4.25.1
# module load datasets/2.11.0
# module load torch_scatter/2.0.9
# module load torch_geometric/2.0.4
# module load torch_sparse/0.6.13

# source ~/.conda/envs/torch_env/bin/activate


# echo "Start to train model"
# python train.py \
#     --data data/proc_datas/ \
#     --batch 16 \
#     --name hpt \

# echo "Start to test model"
# python test.py \
#     --name memes-subtask1-hpt \
#     --data data/proc_datas/ \
#     --batch 32

echo "Make predictions"
python predict.py 
echo "Finished successfully"