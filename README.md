# GRPO: Prelude-SFT-Excellent-v0.2 on INTELLECT-3-RL

GRPO with verl on the math, science and code subsets, on Leonardo. Uses Anna's verl environment.

## Run

Clone the repo into your own folder in the project and run everything from there:

```bash
cd /leonardo_work/AIFAC_F02_352/<your folder>
git clone <repo URL> oellm_grpo && cd oellm_grpo
source /leonardo_work/AIFAC_F02_352/anna/verl/.venv/bin/activate
python prepare_data.py
deactivate
```

Trial run first (2 short steps, debug queue):

```bash
EXPERIMENT=trial sbatch --qos=boost_qos_dbg --time=00:30:00 run_grpo.sh \
  trainer.total_training_steps=2 trainer.test_freq=2 trainer.save_freq=-1 \
  data.train_batch_size=8 actor_rollout_ref.rollout.n=4 data.max_response_length=1024 \
  data.val_max_samples=64 actor_rollout_ref.rollout.val_kwargs.n=1 trainer.logger='[console]'
```

If successful, start the real run:

```bash
EXPERIMENT=my_run sbatch run_grpo.sh
```

To resume after time limit, submit again with the same `EXPERIMENT`.

## TODO

- Set up wandb logging
- Add a reward function for the logic subset (currently left out of the data)