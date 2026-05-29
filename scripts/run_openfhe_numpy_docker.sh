#!/usr/bin/env bash
set -euo pipefail

docker build -t openfhe-numpy-ckks:ubuntu24 docker/openfhe-numpy
docker run --rm -it -v "$PWD":/work -w /work openfhe-numpy-ckks:ubuntu24 bash
