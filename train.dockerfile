# start from a pinned version of tensorflow gpu with python 3 on ubuntu 18.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV HOME=/root
WORKDIR $HOME

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# update os package manager, then install prerequisite packages
RUN apt-get update && apt-get install vim -y && apt-get install -y --no-install-recommends git

# install base requirements
COPY requirements.txt $HOME/
RUN pip install -r requirements.txt

# copy data for local testing experiments
COPY data/gyafc_raw/ $HOME/data/gyafc_raw/
COPY data/imagecaption_raw/ $HOME/data/imagecaption/

# copy everything else (excluding stuff specified in .dockerignore)
COPY . $HOME/

# pip install this package so that it is accessible anywhere
RUN pip install --no-deps -e .


