FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG py_version=3
# Validate that arguments are specified
RUN test $py_version || exit 1

# Install python and nginx
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        jq \
        git \
        ssh \
        libsm6 \
        libxext6 \
        libxrender-dev \
        nginx && \
    if [ $py_version -eq 3 ]; \
       then apt-get install -y --no-install-recommends python3.6-dev \
           && ln -s -f /usr/bin/python3.6 /usr/bin/python; \
       else apt-get install -y --no-install-recommends python-dev; fi && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && rm get-pip.py

RUN pip install --upgrade pip

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install cython \
  && pip install -U natsort \
  && pip install -U tqdm \
  && pip install -U pytorchcv \
  && pip install -U albumentations\
  && pip install opencv-python \
  && pip install tensorboard==1.14 \
  && pip install sagemaker-containers

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && pip install -U pip setuptools \
 && python setup.py build_ext install

COPY . .

ENTRYPOINT ["python", "main_publaynet.py"]
