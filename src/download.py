import os, requests, zipfile
from tqdm import tqdm

URLS = {
    "setA": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",
    "setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",
}
RAW_DIR = "data/raw"

def download_and_extract(name, url):
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, f"{name}.zip")
    print(f"\nDownloading {name}...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.join(RAW_DIR, name))
    print(f"{name} ready.")

if __name__ == "__main__":
    for name, url in URLS.items():
        download_and_extract(name, url)
    print("\nAll data downloaded.")
