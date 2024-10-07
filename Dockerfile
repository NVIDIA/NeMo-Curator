FROM rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.10

WORKDIR /opt

ARG FORKED_REPO_URL
ARG CURATOR_COMMIT

RUN conda create -y --name rapids -c conda-forge -c nvidia \
  python=3.10 \
  cuda-cudart \
  libcufft \
  libcublas \
  libcurand \
  libcusparse \
  libcusolver

RUN bash -exu <<EOF
  git clone $FORKED_REPO_URL
  cd NeMo-Curator
  git fetch origin $CURATOR_COMMIT --depth=1
  git checkout $CURATOR_COMMIT
  source activate rapids
  conda install -y -c conda-forge cython pytest setuptools
  pip install --upgrade pip
  pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
EOF
