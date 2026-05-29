FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/openfhe_numpy_venv
ENV VIRTUAL_ENV=/opt/openfhe_numpy_venv
ENV PATH="/opt/openfhe_numpy_venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install openfhe_numpy numpy pandas

WORKDIR /work

CMD ["bash"]
