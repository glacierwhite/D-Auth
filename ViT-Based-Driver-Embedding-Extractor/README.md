## Data Augmentation LDM

## 💡 : Method
<div align="center">
<img width="600" alt="image" src="./figs/vit.png">
</div>

The synthesized data together with the (limited) real data are then used to train the ***ViT-Based Driver Embedding Extractor***, which learns the mapping from the spectrogram of the motion sensor data to embeddings such that the spectrogram embeddings of the same driver are pulled together, while the spectrogram embeddings of distinct drivers are pushed away.

## ⚙ : Setup
First create a new conda environment

    conda env create -f environment.yml
    conda activate ldm

## ☕️ : Training
You should first download the pretrained weights of [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) and put it to `./ckpt/` folder. Then, you can get the initial weights for training by:

    python utils/prepare_weights.py init_local ckpt/v1-5-pruned.ckpt configs/config.yaml ckpt/init.ckpt

The 4 arguments are mode, pretrained SD weights, model configs and output path for the initial weights.

Now, you can train with you own data simply by:

    python src/train/train.py

## 💻 : Generation
You can launch the generation by:

    python src/generate/generate.py
