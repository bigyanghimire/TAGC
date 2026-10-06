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
