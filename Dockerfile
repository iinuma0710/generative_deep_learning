FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

WORKDIR /workspace
COPY ./requirements.txt /workspace

RUN apt-get update && apt-get install -y unzip graphviz curl musescore3
RUN pip install -U pip && pip install -r /workspace/requirements.txt && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126