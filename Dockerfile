# # © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

FROM nvcr.io/nvidia/pyg:24.03-py3

# Alternatively use this
# FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
# Update package lists and install Python3 and python3-pip
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip

# (Optional) Upgrade pip to its latest version
# RUN pip3 install --upgrade pip

WORKDIR /analysis

COPY requirements.txt /analysis/requirements.txt
COPY setup.py /analysis/setup.py

# copy code files
COPY analysis/__init__.py /analysis/analysis/__init__.py
COPY analysis/experiment /analysis/analysis/experiment
COPY service.py /analysis/service.py
COPY gunicorn.conf.py /analysis/gunicorn.conf.py

# if installing from JHUAPL artifactory
#ENV PIP_INDEX_URL=https://artifactory.jhuapl.edu/artifactory/api/pypi/python-remote/simple
RUN pip install -e .  --default-timeout 1000

# fixes potential dnabert issue if you are running inferences on a H100 instead of a A100
# RUN pip uninstall -y triton

RUN chmod +x /analysis/analysis/experiment/test_sequences.py

CMD ["sleep", "infinity"]