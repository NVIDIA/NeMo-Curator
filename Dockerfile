FROM python:3.10

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
