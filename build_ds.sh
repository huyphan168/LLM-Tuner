mkdir -p /tmp && \
cd /tmp && \
git clone https://github.com/microsoft/DeepSpeed && \
cd DeepSpeed && \
pip install -r requirements/requirements-dev.txt && \
pip install -r requirements/requirements.txt && \
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=0 pip install -v .