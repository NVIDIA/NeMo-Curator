FROM python:3.10

WORKDIR /opt

ARG CURATOR_COMMIT

RUN <<"EOF" bash -exu
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
git checkout $CURATOR_COMMIT
pip install cython pytest setuptools
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
EOF
