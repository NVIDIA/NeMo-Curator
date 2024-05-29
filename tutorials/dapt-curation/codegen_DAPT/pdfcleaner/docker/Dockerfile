FROM nvcr.io/nvidia/tritonserver:21.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt update --fix-missing
RUN pip install --upgrade pip
RUN apt install software-properties-common -y
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt update
RUN apt install cmake libfreetype6-dev libfontconfig1-dev libnss3-dev libtiff-dev libboost-dev qtbase5-dev libopenjp2-7-dev liblcms2-dev libcairo-dev libgtk-3-dev libgirepository1.0-dev -y
RUN sed -i"" 's/mozilla\/DST_Root_CA_X3.crt/!mozilla\/DST_Root_CA_X3.crt/' /etc/ca-certificates.conf \
    && dpkg-reconfigure -fnoninteractive ca-certificates && update-ca-certificates
RUN git clone --branch poppler-23.04.0 https://anongit.freedesktop.org/git/poppler/poppler.git /poppler \
    && cd /poppler && mkdir build && cd build \
    && cmake -D ENABLE_UTILS=ON .. && make -j && make install && export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    && cd / && rm -rf /poppler
RUN pip install ftfy
RUN pip install jupyterlab
