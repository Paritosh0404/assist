#!/usr/bin/env bash
sudo apt-get update && sudo apt-get -y upgrade

# Core dev libraries
sudo apt-get -y install python3-venv python3-pip \
    libopenblas-dev libblas-dev libhdf5-dev libjpeg-dev \
    espeak-ng libespeak1

python3 -m venv ~/venv && source ~/venv/bin/activate

# Ultralytics + specific torch build for ARMv7
pip install --no-cache-dir torch==2.5.0+cpu \
            torchvision==0.20.0+cpu \
            --extra-index-url https://download.pytorch.org/whl/cpu

pip install ultralytics==8.3.70 ncnn opencv-python \
            gpiozero pyttsx3 RPi.GPIO numpy
