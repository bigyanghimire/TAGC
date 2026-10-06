Run `build_apptainer.sh`
and then the command below to prepare data.
Changed files: `nanoGPT/data/openwebtext/prepare.py`

num_proc = 16
LOCAL_DIR = os.path.abspath("/share/bigyang/local")
os.environ["HF_HOME"] = LOCAL_DIR
## Prepare data
```apptainer exec --userns --cleanenv \
  --bind "$PWD:/workspace/TAGC" \
  --bind /share/bigyang:/share/bigyang \
  --pwd /workspace/TAGC build/tagc.sif \
  /workspace/TAGC/.venv/bin/python nanoGPT/data/openwebtext/prepare.py```


  ## Apptainer setup

Build on an internet-connected node:

```bash
cd TAGC
bash build_apptainer.sh
```

The image provides CUDA 12.4.1, Python 3.12 and build tools. All Python
packages, including PyTorch 2.4.1, LHC and TAGC, are installed with pip in `.venv`.
The script applies `patches/torch.diff` to PyTorch in that venv.
The CUDA extension defaults to A100 (`TORCH_CUDA_ARCH_LIST=8.0`), as in
BigBit; set that variable before building for another GPU.
Use `bash build_apptainer.sh --install-only` to reuse the image.

Prepare OpenWebText on the internet-connected node:

```bash
apptainer exec --userns --cleanenv \
  --bind "$PWD:/workspace/TAGC" --bind /share/bigyang:/share/bigyang \
  --pwd /workspace/TAGC \
  build/tagc.sif /workspace/TAGC/.venv/bin/python nanoGPT/data/openwebtext/prepare.py
```

Run the following on each of the two GPU nodes with the same shared checkout.
Replace `<IP-of-the-master-rank>` with the first node's address:

```bash
apptainer exec --userns --cleanenv --nv \
  --bind "$PWD:/workspace/TAGC" --pwd /workspace/TAGC/nanoGPT \
  build/tagc.sif /workspace/TAGC/.venv/bin/python -m torch.distributed.run \
  --rdzv-backend=c10d --rdzv-endpoint=<IP-of-the-master-rank> \
  --nnodes=2 --nproc-per-node=1 train.py config/train_gpt2.py
```
