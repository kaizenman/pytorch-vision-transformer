import os

import requests
from pathlib import Path
from zipfile import ZipFile

def download_data(url, file_name, dest):
  dest_path = Path(dest)
  if dest_path.is_dir():
    print(f'{dest_path} directory exist. Skipping download.')
    return

  print(f'{dest_path} directory is missing. Creating...')
  dest_path.mkdir(parents=True, exist_ok=True)
  response = requests.get(url)
  with open(dest_path / file_name, mode='wb') as f:
    print(f'downloading {dest_path / file_name}...')
    f.write(response.content)
    print(f'downloaded.')
  with ZipFile(dest_path / file_name) as zip:
    print(f'extracting {dest_path / file_name}...')
    zip.extractall(dest_path)
    print(f'extracted.')
  print(f'removing {dest_path / file_name}...')
  os.remove(dest_path / file_name)
  print(f'removed')
  return

