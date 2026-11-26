#!/bin/bash
# Installs upstream verl (with vLLM and FSDP) in its own uv environment, separate from any other project.
# Run on a LOGIN node (needs internet), inside tmux; takes 15-30 minutes.
# Usage: BASE=/leonardo_work/<project>/<you> bash install_verl.sh
set -euo pipefail

BASE=${BASE:?set BASE to your folder in the project, e.g. BASE=/leonardo_work/<project>/<you>}
mkdir -p "$BASE"/{cache,models,oellm_grpo/logs}
export UV_CACHE_DIR="$BASE/cache/uv" UV_PYTHON_INSTALL_DIR="$BASE/cache/uv-python" HF_HOME="$BASE/cache/hf"
export XDG_CACHE_HOME="$BASE/cache" PIP_CACHE_DIR="$BASE/cache/pip"   # keep everything out of the small $HOME quota
# uv's own Python doesn't find the system certificates; some NVIDIA packages download their wheels with it
export SSL_CERT_FILE=${REQUESTS_CA_BUNDLE:-/etc/pki/tls/certs/ca-bundle.crt}   # REQUESTS_CA_BUNDLE is preset (read-only) on Leonardo

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$BASE"
[ -d verl ] || git clone https://github.com/verl-project/verl.git
cd verl
git log -1 --format="verl commit %h (%cd)"

# verl's documented custom-environment setup (Python 3.12, its locked versions), with the vLLM and FSDP extras.
# uv downloads its own Python 3.12 rather than using an active conda env.
rm -rf .venv
uv sync --frozen --all-packages --extra vllm --extra fsdp --python 3.12 --managed-python
source .venv/bin/activate
uv pip install math-verify   # used by the math reward

python - <<'PY'
import torch, vllm, transformers, verl
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| verl", verl.__version__)
PY
echo "Environment: $BASE/verl/.venv"