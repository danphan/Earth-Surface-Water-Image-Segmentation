import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")

# If the image folder doesn't exist, download it and prepare it... 
if data_path.is_dir():
    print(f"{data_path} directory exists.")
else:
    print(f"Did not find {data_path} directory, creating one...")
    data_path.mkdir(parents=True, exist_ok=True)

# Download data
with open(data_path / "dset-s2.zip", "wb") as f:
    request = requests.get('https://zenodo.org/record/5205674/files/dset-s2.zip')
    f.write(request.content)

# Unzip data
with zipfile.ZipFile(data_path / "dset-s2.zip", "r") as zip_ref:
    print("Unzipping data...") 
    zip_ref.extractall(data_path)

# Remove zip file
os.remove(data_path / "dset-s2.zip")
