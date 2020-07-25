FROM gcr.io/blueshift-playground/blueshift:gpu
RUN [ $(getent group 1001) ] || groupadd --gid 1001 1001
RUN useradd --no-log-init --no-create-home -u 1001 -g 1001 --shell /bin/bash esowc
RUN mkdir -m 777 /usr/app /.creds /home/esowc
ENV HOME=/home/esowc
WORKDIR /usr/app
USER 1001:1001
COPY --chown=1001:1001 environment.yml /usr/app
RUN /bin/bash -c "/opt/conda/bin/conda env update --quiet --name caliban --file environment.yml && /opt/conda/bin/conda clean -y -q --all"
COPY --chown=1001:1001 . /usr/app/.
ENTRYPOINT ["fish"]
