#!/bin/bash
#SBATCH --job-name=build-flashattn-verl
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Builds flash-attn 2.8.3 for the upstream verl env ($BASE/verl/.venv), A100 only.
# Resumable: if it hits the 30-min limit, sbatch it again; objects in build/ are reused.
set -eo pipefail

BASE=/leonardo_work/AIFAC_F02_352/anna
source $BASE/verl/.venv/bin/activate

module load gcc/12.2.0
export CUDA_HOME=$BASE/cuda-13.0 PATH=$BASE/cuda-13.0/bin:$PATH   # own toolkit: Leonardo's modules stop at 12.6, torch needs 13
export LD_LIBRARY_PATH=$(ls -d $BASE/cuda-compat/usr/local/cuda-*/compat | tail -1):$LD_LIBRARY_PATH
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
NVCC_CUDA=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "torch $(python -c 'import torch; print(torch.__version__)') (CUDA $TORCH_CUDA), nvcc $NVCC_CUDA"
[ "${TORCH_CUDA%%.*}" = "${NVCC_CUDA%%.*}" ] || { echo "nvcc and torch CUDA major versions differ"; exit 1; }

export FLASH_ATTENTION_FORCE_BUILD=TRUE   # never fetch the prebuilt wheel
export FLASH_ATTN_CUDA_ARCHS=80           # A100 only
export NVCC_THREADS=1 MAX_JOBS=28         # ~4-6 GB RAM per nvcc job
export TMPDIR=$BASE/tmp && mkdir -p $TMPDIR

# Own copy of the source, so objects built for the other envs' torch are never reused
SRC=$BASE/src/flash-attention-verl
[ -d $SRC ] || rsync -a --exclude build --exclude dist $BASE/src/flash-attention/ $SRC/
cd $SRC
python setup.py bdist_wheel               # incremental: resumes from build/
uv pip install --no-deps --reinstall dist/flash_attn-*.whl

python -c "import torch, flash_attn; from flash_attn import flash_attn_func; \
q=torch.randn(1,128,8,64,device='cuda',dtype=torch.bfloat16); \
print('flash_attn', flash_attn.__version__, flash_attn_func(q,q,q).shape)"
python -c "import torch, vllm, verl, flash_attn; \
print('torch', torch.__version__, '| vllm', vllm.__version__, '| verl', verl.__version__, '| cuda', torch.cuda.is_available())"