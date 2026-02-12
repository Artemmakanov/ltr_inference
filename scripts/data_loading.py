import pathlib
import requests

BASE = "https://dlsun.github.io/pods/data/ml-1m"  # [web:12]

FILES = ["ratings.dat", "movies.dat", "users.dat"]

def download_movielens_1m_alt(dest_dir: str = "ml-1m-alt") -> pathlib.Path:
    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        url = f"{BASE}/{name}"
        out = dest / name
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

    return dest

if __name__ == "__main__":
    path = download_movielens_1m_alt()
    print("Saved to", path.resolve())
