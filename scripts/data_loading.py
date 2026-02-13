import pathlib
import requests
from tqdm import tqdm  # pip install tqdm

BASE = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"  # Direct stable link

def download_movielens_10m(dest_dir: str = "ml-10m") -> pathlib.Path:
    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    url = BASE
    zip_path = dest / "ml-10m.zip"
    out_dir = dest / "ml-10m"
    
    if not out_dir.exists():
        print("Downloading ml-10m.zip with progress...")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="ml-10m.zip",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
        
        print("Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest)
        zip_path.unlink()
    
    return out_dir

if __name__ == "__main__":
    path = download_movielens_10m()
    print("Saved to", path.resolve())
