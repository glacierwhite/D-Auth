## Data Augmentation LDM

## 💡 : Method
<div align="center">
<img width="600" alt="image" src="./figs/vit.png">
</div>

The synthesized data together with the (limited) real data are then used to train the ***ViT-Based Driver Embedding Extractor***, which learns the mapping from the spectrogram of the motion sensor data to embeddings such that the spectrogram embeddings of the same driver are pulled together, while the spectrogram embeddings of distinct drivers are pushed away.

## ⚙ : Setup
First create a new conda environment

    conda env create -f environment.yml
    conda activate vit

## ☕️ : Training
You can train the model by:

    python src/train.py

## 💻 : Test
You can test the model by:

    python src/test.py
