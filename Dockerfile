ARG GPU=1
ARG CUDA_VERSION=10.1

FROM gcr.io/blueshift-playground/blueshift:gpu
RUN apt update && apt install -qq -o=Dpkg::Use-Pty=0 fish -y
RUN [ $(getent group 1001) ] || groupadd --gid 1001 1001
RUN useradd --no-log-init --no-create-home -u 1001 -g 1001 --shell /bin/bash esowc
RUN mkdir -m 777 /usr/app /.creds /home/esowc
ENV HOME=/home/esowc
WORKDIR /usr/app
USER 1001:1001
COPY --chown=1001:1001 environment.yml /usr/app
RUN /bin/bash -c "/opt/conda/bin/conda env update --quiet --name caliban --file environment.yml && /opt/conda/bin/conda clean -y -q --all"
COPY --chown=1001:1001 . /usr/app/.

RUN echo "Installing Apex"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /usr/app
ENTRYPOINT ["fish"]
