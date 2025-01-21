# Catalyst via Bayesian Optimization


### Installation
get mamba (better than conda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

```bash
mamba create -n boc python=3.10 -y
mamba activate boc
pip install pyscf numpy==1.24.4 plotly kaleido scipy scikit-learn matplotlib==3.8.4 seaborn black tqdm joblib einops pandas ipykernel botorch
mamba install cupy=13.3
pip install torch
# pip install jax flax
```

### Run the bayesian optimization
```bash
mamba activate boc
python train.py
```

