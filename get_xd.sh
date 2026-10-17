#!/bin/bash
#PBS -N xd_download
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -j oe
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-lvmae}
XD_ROOT=${XD_ROOT:-$HOME/XD_Violence}
mkdir -p $XD_ROOT && cd $XD_ROOT
df -h $XD_ROOT | tail -1
python - <<'PY'
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
REPO = "jherng/xd-violence"
root = os.getcwd()
files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
         if (f.startswith("data/video/") and f.endswith(".mp4")) or (f.count("/") == 1 and f.startswith("data/") and f.endswith(".txt"))]
todo = [f for f in files if not os.path.isfile(os.path.join(root, f))]
print(f"repo files {len(files)}, already here {len(files) - len(todo)}, to download {len(todo)}", flush=True)

def get(f):
    err = None
    for attempt in range(4):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=root, force_download=attempt > 0)
            return f, None
        except Exception as e:
            err = e
            time.sleep(5 * (attempt + 1))
    return f, err

failed = []
with ThreadPoolExecutor(8) as ex:
    for i, fut in enumerate(as_completed([ex.submit(get, f) for f in todo]), 1):
        f, err = fut.result()
        if err is not None:
            failed.append(f)
            print(f"[FAILED] {f}: {err!r}", flush=True)
        if i % 50 == 0:
            print(f"{i}/{len(todo)} done, failed {len(failed)}", flush=True)
print(f"download finished, failed {len(failed)}", flush=True)
PY
echo "test videos : $(find data/video -path '*test*' -name '*.mp4' | wc -l)  (expect 800)"
echo "all videos  : $(find data/video -name '*.mp4' | wc -l)  (expect 4754)"
du -sh data/video; df -h $XD_ROOT | tail -1
