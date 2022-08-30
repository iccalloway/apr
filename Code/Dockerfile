FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV PROJECT_HOME="/stt"
COPY . $PROJECT_HOME
WORKDIR $PROJECT_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    python3.8 \
    python3.8-dev \
    python3-pip \
    vim

# Install dependencies first, advantage layer caching
RUN python3.8 -m pip install -U pip setuptools 
RUN python3.8 -m pip install -r $PROJECT_HOME/requirements.txt 

WORKDIR $PROJECT_HOME
