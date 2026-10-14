Run `build_apptainer.sh`
and then the command below to prepare data.
Changed files: `nanoGPT/data/openwebtext/prepare.py`

num_proc = 16

The data preparer uses `HF_HOME` from the environment. When running from this
submodule, read the cache path from the parent framework's `config.json` and
forward it through Apptainer's clean environment:

## Prepare data

```bash
export APPTAINERENV_HF_HOME="$(python -c 'import json, os; path = json.load(open("../config.json"))["hf_cache_dir"]; print(os.path.abspath(os.path.expanduser(os.path.expandvars(path))))')"
apptainer exec --userns --cleanenv \
  --bind "$PWD:/workspace/TAGC" \
  --bind /share/bigyang:/share/bigyang \
  --pwd /workspace/TAGC build/tagc.sif \
  /workspace/TAGC/.venv/bin/python nanoGPT/data/openwebtext/prepare.py
```

Adjust the shared-storage bind mount to match the cache path on your cluster.
