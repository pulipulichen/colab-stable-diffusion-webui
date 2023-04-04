#!/bin/bash

FILE="/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/ready.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE not exists."

    
    apt -y update -qq
    wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb
    wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb
    apt install -qq libunwind8-dev
    dpkg -i *.deb
    env LD_PRELOAD=libtcmalloc.so
    rm *.deb

    apt -y install -qq aria2
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U
    pip install -q xformers==0.0.16 triton==2.0.0 -U

    #git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui"
    cd "/content/drive/My Drive/Colab Notebooks/"
    git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui
    git clone https://huggingface.co/embed/negative "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/embeddings/negative"
    git clone https://huggingface.co/embed/lora "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Lora/positive"
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/ESRGAN" -o 4x-UltraSharp.pth
    wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/scripts/run_n_times.py"
    git clone https://github.com/deforum-art/deforum-for-automatic1111-webui "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui"
    git clone https://github.com/camenduru/stable-diffusion-webui-images-browser "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser"
    git clone https://github.com/camenduru/stable-diffusion-webui-huggingface "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface"
    git clone https://github.com/camenduru/sd-civitai-browser "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-civitai-browser"
    git clone https://github.com/kohya-ss/sd-webui-additional-networks "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-additional-networks"
    git clone https://github.com/Mikubill/sd-webui-controlnet "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet"
    git clone https://github.com/camenduru/openpose-editor "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/openpose-editor"
    git clone https://github.com/jexom/sd-webui-depth-lib "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-depth-lib"
    git clone https://github.com/hnmr293/posex "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/posex"
    git clone https://github.com/camenduru/sd-webui-tunnels "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-tunnels"
    git clone https://github.com/etherealxx/batchlinks-webui "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/batchlinks-webui"
    git clone https://github.com/camenduru/stable-diffusion-webui-catppuccin "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin"
    git clone https://github.com/KohakuBlueleaf/a1111-sd-webui-locon "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/a1111-sd-webui-locon"
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg"
    git clone https://github.com/ashen-sensored/stable-diffusion-webui-two-shot "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot"
    cd "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui"
    git reset --hard

    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_canny-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_canny-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_depth-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_depth-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_hed-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_hed-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_mlsd-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_mlsd-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_normal-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_normal-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_openpose-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_openpose-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_scribble-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_scribble-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/control_seg-fp16.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o control_seg-fp16.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/hand_pose_model.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose" -o hand_pose_model.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/body_pose_model.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose" -o body_pose_model.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/dpt_hybrid-midas-501f0c75.pt -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/midas" -o dpt_hybrid-midas-501f0c75.pt
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_large_512_fp32.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/mlsd" -o mlsd_large_512_fp32.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_tiny_512_fp32.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/mlsd" -o mlsd_tiny_512_fp32.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/network-bsds500.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/hed" -o network-bsds500.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/upernet_global_small.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/uniformer" -o upernet_global_small.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_style_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_style_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_sketch_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_sketch_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_seg_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_seg_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_openpose_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_openpose_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_keypose_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_keypose_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_depth_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_depth_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_color_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_color_sd14v1.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_canny_sd14v1.pth -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/sd-webui-controlnet/models" -o t2iadapter_canny_sd14v1.pth

    # Models
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/OrangeMixs/resolve/main/AOM3.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Stable-diffusion" -o AOM3.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/OrangeMixs/resolve/main/AOM3A1.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Stable-diffusion" -o AOM3A1.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/OrangeMixs/resolve/main/AOM3A2.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Stable-diffusion" -o AOM3A2.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/OrangeMixs/resolve/main/AOM3A3.safetensors -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Stable-diffusion" -o AOM3A3.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -d "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/models/Stable-diffusion" -o orangemix.vae.pt

    sed -i -e '''/    prepare_environment()/a\    os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' ''/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py''""")''' "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/launch.py"
    sed -i -e 's/fastapi==0.90.1/fastapi==0.89.1/g' "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/requirements_versions.txt"

    mkdir -p "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui/models"
    touch "/content/drive/My Drive/Colab Notebooks/stable-diffusion-webui/"

    touch $FILE
fi

python launch.py --listen --xformers --enable-insecure-extension-access --theme dark --gradio-queue --multiple