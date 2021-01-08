# build Terastitcher binaries
FROM ubuntu:bionic AS terastitcher

RUN apt update
RUN apt install -y git build-essential cmake libscalapack-openmpi-dev openmpi-bin
RUN git clone https://github.com/abria/TeraStitcher
RUN mkdir build-terastitcher
RUN cd build-terastitcher && \
    cmake -DWITH_UTILITY_MODULE_teraconverter=ON -DWITH_UTILITY_MODULE_mdatagenerator=ON -DWITH_UTILITY_MODULE_volumeconverter=ON -DWITH_UTILITY_MODULE_mergedisplacements=ON ../TeraStitcher/src && \
    make -j `nproc` && \
    # need ownership of /usr/local to install without sudo
    # chown -R ${USER}:${USER} /usr/local/ && \
    make install

# now install CloudReg
FROM intelpython/intelpython3_full

# create environment variables for credentials
ENV SSH_KEY_PATH=/run/secrets/ssh_key
ENV CV_CRED_PATH=/run/secrets/cloudvolume_credentials

# install CloudReg
RUN git clone https://github.com/neurodata/CloudReg.git
RUN cd CloudReg && \
    pip install -r requirements.txt && \
    pip install --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/v2.0rc1

# installed terastitcher binaries are in /usr/local
# we want terastitcher, teraconverter, mergedisplacements, mdatagenerator
COPY --from=terastitcher /usr/local/bin/terastitcher /usr/local/bin/teraconverter /usr/local/bin/mdatagenerator /usr/local/bin/mergedisplacements /usr/local/bin/

ADD entrypoint.sh entrypoint.sh

ENTRYPOINT [ "/bin/bash", "entrypoint.sh" ]
CMD [ "-h" ]
# ENTRYPOINT [ "python", "CloudReg/scripts/colm_pipeline.py" ]