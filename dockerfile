FROM gcr.io/tensorflow/tensorflow
RUN apt-get update && apt-get install -y git-core tmux
RUN git clone https://github.com/rfcoles1/Chemical-Kinetics.git ./notebooks
WORKDIR "/notebooks"
RUN pwd
RUN pip install -r ./requirements.txt

FROM confluent/postgres-bw:0.1
RUN apt-get install -y vim

CMD ["/run_jupyter.sh"]
