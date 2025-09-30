#!/bin/bash
#SBATCH --array=1-1%1
#SBATCH --job-name=dataset
#SBATCH --nodes=1                # nombre de noeuds
#SBATCH --ntasks=1               # nombre total de tâches sur tous les nœuds
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --output=hs_slurm/dcv_hist/out/slurm-%A_%a.txt
#SBATCH --error=hs_slurm/dcv_hist/err/slurm-%A_%a.txt
#SBATCH --mail-type=ALL
#SBATCH --requeue

# export TMPDIR=/scratch/<project>/tmp


BATCH_HIST="batch.txt"



CMD=$"srun sh 1_download/run_all.sh"

echo "start"
ml python/3.12
ml cuda/12.3
source .env/bin/activate
echo "$SLURM_ARRAY_TASK_ID|$CMD" >> $BATCH_HIST
$CMD
deactivate
echo "end"