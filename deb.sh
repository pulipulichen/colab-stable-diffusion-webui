#!/bin/bash

FILE="/content/drive/MyDrive/Colab Notebooks/stable-diffusion-webui-deb/ready.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE not exists."
    mkdir -p "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui-deb/"
    cd "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui-deb/"
    wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb

    touch "$FILE"
fi

cd "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui-deb/"

apt -y update -qq

apt install -qq libunwind8-dev
dpkg -i *.deb
env LD_PRELOAD=libtcmalloc.so

apt -y install -qq aria2
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U
pip install -q xformers==0.0.16 triton==2.0.0 -U

#rm *.deb