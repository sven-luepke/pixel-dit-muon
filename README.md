# Training Diffusion Transformers with Muon

Code for the blog post [Training Diffusion Transformers with Muon](https://sven-luepke.github.io/blog/2026-05-31-dit-muon/)

### Setup
```
conda env create -f environment.yml
conda activate pixel-dit-muon
hf auth login --token <your_huggingface_token>
wandb login <your_wandb_api_key>
```

### Training DiT-L
With Adam:
```
python train.py --steps 200000 
```
With Muon:
```
python train.py --steps 200000 --muon
```
