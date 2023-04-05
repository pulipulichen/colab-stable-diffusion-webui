#!/bin/bash

FILE="/content/drive/MyDrive/Colab Notebooks/stable-diffusion-webui/models/Lora/Japanese-doll-likeness.safetensors"
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE not exists."
    # https://huggingface.co/AnonPerson/ChilloutMix/resolve/main/Japanese-doll-likeness.safetensors
    wget https://huggingface.co/AnonPerson/ChilloutMix/resolve/main/Japanese-doll-likeness.safetensors -O "$FILE"
fi