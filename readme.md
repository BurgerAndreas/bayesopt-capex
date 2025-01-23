# Catalyst discovery with Bayesian Optimization


### Installation
get mamba (better than conda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

```bash
mamba create -n boc python=3.10 -y
mamba activate boc
pip install pyscf numpy==1.24.4 plotly kaleido scipy scikit-learn matplotlib==3.8.4 seaborn black tqdm joblib einops pandas ipykernel jupyter 


# if you are on mac first run this
conda install pytorch torchvision -c pytorch

# if you are on windows or linux
pip install torch

# install ax after torch, so it is linked against MKL for 10x speedup
pip install ax-platform==0.4.3
```

### Run the bayesian optimization
```bash
mamba activate boc
python train.py
```

