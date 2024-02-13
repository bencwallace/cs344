FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y build-essential cmake unzip wget

RUN wget https://github.com/opencv/opencv/archive/3.4.16.zip -O /tmp/opencv.zip && \
    unzip /tmp/opencv.zip -d /tmp && \
    cmake -S /tmp/opencv-3.4.16 -B /tmp/opencv-3.4.16/build && \
    make -C /tmp/opencv-3.4.16/build -j && \
    make -C /tmp/opencv-3.4.16/build install
