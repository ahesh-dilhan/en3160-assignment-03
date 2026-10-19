"""Run all experiments once and preserve their genuine console transcript."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PYTHONUNBUFFERED','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
with (ROOT/'results'/'training_transcript.txt').open('w',encoding='utf-8') as log:
    command=[sys.executable,'-u','-m','src.experiments','--task','all']
    log.write('Command: python -u -m src.experiments --task all\n')
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace')
    for line in process.stdout:
        print(line,end='',flush=True)
        log.write(line);log.flush()
    status=process.wait()
    if status:raise SystemExit(status)
