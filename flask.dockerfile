# start from a pinned version of tensorflow gpu with python 3 on ubuntu 18.04
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
#FROM floydhub/dl-docker:gpu
FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV HOME=/root
WORKDIR $HOME

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# update os package manager, then install prerequisite packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git

# install base requirements
COPY requirements.txt $HOME/
RUN pip install -r requirements.txt

# install api-specific requirements
COPY http_api/requirements.txt $HOME/http_api/requirements.txt
RUN pip install -r http_api/requirements.txt

# copy model, config each style
COPY checkpoints/del_and_ret-formal $HOME/checkpoints/del_and_ret-formal
COPY checkpoints/del_and_ret-romantic $HOME/checkpoints/del_and_ret-romantic

# copy vocab, attribute vocab, original text files for each style
COPY data/gyafc_subword_encoded_10000/ $HOME/data/gyafc_subword_encoded_10000/
COPY data/imagecaption/ $HOME/data/imagecaption/

# copy everything else (excluding stuff specified in .dockerignore)
COPY . $HOME/

# pip install this package so that it is accessible anywhere
RUN pip install --no-deps -e .

# make a non-root user group and add a user
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

# give user group access to home directory
RUN chown 1001:1001 $HOME

USER appuser 

EXPOSE 5000

# start the flask app
ENV FLASK_APP=http_api/flask_app.py
#ENV FLASK_ENV=development
CMD ["flask", "run", "--host=0.0.0.0"]



