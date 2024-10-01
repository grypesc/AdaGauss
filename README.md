# AdaGauss repository

This repostory contains code for NeurIPS 2024 paper on Continual Learning: **Task-recency bias strikes back: Adapting covariances in Exemplar-Free Class Incremental Learning** (<https://arxiv.org/abs/2409.18265>). The repository is based on FACIL benchmark <https://github.com/mmasana/FACIL>.

We consider exemplar free class incremental scenario, where we train feature extractor on all tasks using cross-entropy, feature distillation through a neural projector and anti-collapse loss. We represent each class as Gaussian distribution in the latent space. After each task we transform these distributions from the old model's latent space to the new using an auxilary neural network.

### Setup
Create virtual environment and install dependencies:
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install requirements.txt
```

Reproduce experiments using scripts in ```scripts``` directory:
```bash
bash scripts/cifar-10x10.sh
```
