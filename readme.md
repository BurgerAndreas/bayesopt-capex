# Catalyst discovery with Bayesian Optimization


### Installation
get mamba (better than conda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

```bash
# delete the conda environment
conda remove -n boc --all -y
mamba create -n boc python=3.10 -y
mamba activate boc
# ==0.59.0 
pip install numpy==1.24.4 numba plotly dash table kaleido scipy scikit-learn matplotlib pyarrow umap-learn seaborn black tqdm joblib einops pandas ipykernel nbformat 

pip install torch
```

### Run the bayesian optimization
```bash
mamba activate boc
python train.py
```

### Plotting

```bash
mamba activate boc
python analysis/dashboard_beliefs.py
# http://127.0.0.1:8050
```
