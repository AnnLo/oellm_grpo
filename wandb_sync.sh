#!/bin/bash
#SBATCH --job-name=wandb-sync-grpo
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
# Uploads the offline wandb runs in logs/ to wandb.ai every 10 min and resubmits itself before the time limit.
# Usage: sbatch wandb_sync.sh     Stop: scancel -n wandb-sync-grpo
source /leonardo_work/AIFAC_F02_352/anna/verl/.venv/bin/activate
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

end=$((SECONDS + 3*3600 + 45*60))
while [ $SECONDS -lt $end ]; do
  wandb sync --include-offline logs/*/wandb/offline-run-*
  sleep 600
done
sbatch wandb_sync.sh