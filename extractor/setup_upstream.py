"""Fetch the pinned LSeg source needed by the standalone checkpoint loader."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
target = root/'src/lseg_feature_extraction'
commit = '60afac405fe146c1c9df01ec3ea1baee39d5ee3b'
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/pengsongyou/lseg_feature_extraction.git', str(target)], check=True)
subprocess.run(['git', '-C', str(target), 'checkout', '--detach', commit], check=True)
print('Pinned LSeg source:', commit)
