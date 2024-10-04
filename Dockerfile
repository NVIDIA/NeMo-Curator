FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /opt

ARG FORKED_REPO_URL
ARG CURATOR_COMMIT

RUN <<"EOF" bash -exu
git clone $FORKED_REPO_URL
cd NeMo-Curator
git fetch origin $CURATOR_COMMIT --depth=1
git checkout $CURATOR_COMMIT
pip install cython pytest setuptools pip --upgrade
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
EOF
