FROM python:3.8-slim

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    make automake gcc g++ subversion python3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY . ./brainlit/

RUN pip install --upgrade pip setuptools && \
    cd brainlit && pip install -e .