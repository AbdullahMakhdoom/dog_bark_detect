FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

COPY . /app/

RUN apt-get update

RUN set -xe  && apt-get install -y python3 python3-pip

RUN apt-get update \
        && apt-get install gcc libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev alsa-utils -y 

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install pytorch-lightning

CMD ["python3", "real_time_inference.py"]