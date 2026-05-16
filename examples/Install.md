curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12

source .venv/bin/activate

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

uv pip install -e .

uv pip install -e ".[notebooks]"

uv pip install -e ".[train,dev]"

uv pip install einops ninja && uv pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu130

apt-get install -y build-essential g++ ninja-build

uv pip install git+https://github.com/ronghanghu/cc_torch.git --no-build-isolation


uv pip install modelscope

modelscope download --model facebook/sam3 --local_dir ./checkpoint/sam3/