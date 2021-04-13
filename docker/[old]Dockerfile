FROM python:3.8-slim
#RUN pip install --no-cache-dir matplotlib pandas numpy

WORKDIR /usr/src/app

#RUN apt-get update \
#    && apt-get install -y --no-install-recommends git \
#    && apt-get purge -y --auto-remove \
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    make automake gcc g++ subversion python3-dev \
    && rm -rf /var/lib/apt/lists/*


# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev

# RUN apk update && apk upgrade && \
#     apk add --no-cache bash git openssh

# RUN git clone https://github.com/neurodata/brainlit.git

RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir brainlit


# create credentials file for aws
# RUN chmod +x .aws.sh && \ 
#     source .aws.sh

