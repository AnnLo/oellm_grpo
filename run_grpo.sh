#!/bin/bash
#SBATCH --job-name=oellm-grpo
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# GRPO with upstream verl on INTELLECT-3-RL (math, science, code, logic).
# Usage: sbatch run_grpo.sh [hydra overrides]. Resubmitting with the same EXPERIMENT resumes from the last checkpoint.
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_352/anna
DIR=${SLURM_SUBMIT_DIR:-$PWD}
MODEL=/leonardo_scratch/fast/AIFAC_F02_352/models/Prelude-SFT-Excellent-v0.2
EXPERIMENT=${EXPERIMENT:-grpo_prelude_9b}

cd $DIR
mkdir -p logs/$EXPERIMENT
exec > >(tee -a logs/$EXPERIMENT/training.log) 2>&1

source $BASE/verl/.venv/bin/activate
module load gcc/12.2.0
export LD_LIBRARY_PATH=$(ls -d $BASE/cuda-compat/usr/local/cuda-*/compat):$LD_LIBRARY_PATH  # newer CUDA than the driver
export HF_HUB_OFFLINE=1 HF_HOME=$BASE/cache/hf XDG_CACHE_HOME=$BASE/cache   # no internet on compute nodes
export WANDB_MODE=offline WANDB_DIR=$DIR/logs/$EXPERIMENT WANDB_API_KEY=$(cat ~/.wandb_api_key)
unset ROCR_VISIBLE_DEVICES   # set by Slurm; verl refuses it next to CUDA_VISIBLE_DEVICES
# flash-attn is self-built (build_flashattn.sh): verl's prebuilt wheel needs a newer glibc than Leonardo has

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$DIR/data/intellect3/train.parquet \
  data.val_files=$DIR/data/intellect3/test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=5120 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  reward.custom_reward_function.path=$DIR/reward.py \
  trainer.logger='[console,wandb]' \
  trainer.project_name=oellm-grpo \
  trainer.experiment_name=$EXPERIMENT \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_training_steps=300 \
  trainer.val_before_train=True \
  trainer.test_freq=25 \
  trainer.save_freq=25 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=$DIR/checkpoints/$EXPERIMENT \
  ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK \
  "$@"