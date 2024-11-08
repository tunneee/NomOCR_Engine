# FROM ubuntu:latest
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

LABEL maintainer="tuonghh@fpt.com"


# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


    RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
# Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH

# SHELL ["/bin/bash", "-c"]

# RUN conda init bash \
#     && . ~/.bashrc \
#     && conda create --name nomocr python=3.10 \
#     && conda activate nomocr \
    # && pip install ipython


# RUN conda init bash \
#     && . ~/.bashrc \
#     && conda create --name nomocr python=3.10 \
#     && source activate base \
#     && conda activate nomocr \

# Install dependencies

# RUN conda install nvidia/label/cuda-11.8.0::cuda-toolkit -c nvidia/label/cuda-11.8.0 --yes
# RUN conda install -c conda-forge cudnn --yes

# ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/requirements.txt


RUN pip install -r /tmp/requirements.txt
# RUN pip uninstall nvidia_cublas_cu11
COPY . /app

WORKDIR /app

# Run Uvicorn 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "8"]

# RUN echo "source activate env" > ~/.bashrc

# ENV PATH /opt/conda/envs/env/bin:$PATH
