"""Where data lives, on Colab, Kaggle or a laptop.

Every notebook calls ``paths = setup()`` first. Data is never stored in the git repo.

Colab : Google Drive, MyDrive/Navigate/data
Kaggle: /kaggle/working/Navigate/data  (pulled from a Hugging Face dataset repo if HF_DATA_REPO is set)
Laptop: ./data  (or the NAVIGATE_DATA environment variable)
"""
import os


def where():
    if os.path.exists("/kaggle/working"):          # checked first: Kaggle's image also looks like Colab
        return "kaggle"
    if os.path.exists("/content") and "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def get_secret(name, default=None):
    """Read a secret from Colab secrets, Kaggle secrets, or an environment variable."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        if where() == "colab":
            from google.colab import userdata
            return userdata.get(name)
        if where() == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    return default


def setup(city="hyderabad", root=None, hf_repo=None):
    """Create the folders and return a dict of paths."""
    w = where()
    if root is None:
        if os.environ.get("NAVIGATE_DATA"):
            root = os.environ["NAVIGATE_DATA"]
        elif w == "colab":
            if not os.path.exists("/content/drive/MyDrive"):
                from google.colab import drive
                drive.mount("/content/drive")
            root = "/content/drive/MyDrive/Navigate/data"
        elif w == "kaggle":
            root = "/kaggle/working/Navigate/data"
        else:
            root = os.path.abspath("data")
    p = {
        "where": w, "city": city, "root": root,
        "raw": f"{root}/raw",
        "gaz": f"{root}/gazetteer/{city}",      # notebooks 01-03: map tables
        "bench": f"{root}/bench/{city}",        # notebooks 04-07: benchmark tables
        "runs": f"{root}/runs/{city}",          # notebooks 08+: predictions and scores
        "cache": f"{root}/cache/{city}",        # API caches (commercial ones are purged after 30 days)
    }
    for k in ("raw", "gaz", "bench", "runs", "cache"):
        os.makedirs(p[k], exist_ok=True)
    if w == "kaggle":
        import_kaggle_inputs(p)
    hf_repo = hf_repo or os.environ.get("HF_DATA_REPO")
    if w != "colab" and hf_repo and not os.listdir(p["bench"]):
        from huggingface_hub import snapshot_download
        snapshot_download(hf_repo, repo_type="dataset", local_dir=root, token=get_secret("HF_TOKEN"))
    return p


GAZ_FILES = {"buildings.parquet", "landmarks.parquet", "roads.parquet", "areas.parquet", "addr_tagged.parquet", "entrances.parquet",
             "landmark_areas.parquet", "overture_places.parquet", "landmarks_merged.parquet", "real_addresses_overture.parquet",
             "pincode_areas.parquet", "pincode_method.json", "manifest.json", "manifest_02.json"}


def import_kaggle_inputs(p):
    """Kaggle cannot see Google Drive. Files you attach to a Kaggle notebook (Add Input > upload) appear under
    /kaggle/input. This copies every file it recognises into the folder the notebooks expect, whatever
    folder structure the upload had. Files already in place are not overwritten."""
    import shutil
    if not os.path.exists("/kaggle/input"):
        return
    copied = []
    for d, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f in GAZ_FILES:
                dest = p["gaz"]
            elif f.startswith("paraphrase_") or f.endswith(".jsonl"):
                dest = p["cache"]
            elif f.endswith((".parquet", ".json")):
                dest = p["bench"]
            else:
                continue
            if not os.path.exists(os.path.join(dest, f)):
                shutil.copy(os.path.join(d, f), dest)
                copied.append(f"{f} -> {dest}")
    if copied:
        print("copied from /kaggle/input:\n  " + "\n  ".join(copied))


def push_to_hub(paths, repo, folders=("bench", "gazetteer"), private=True):
    """Copy the benchmark and map tables to a Hugging Face dataset repo, so a Kaggle GPU notebook can pull them.
    Needs the secret HF_TOKEN (a token with write access). Not needed if you only use Colab + Google Drive."""
    from huggingface_hub import HfApi
    api = HfApi(token=get_secret("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    for f in folders:
        src = f"{paths['root']}/{f}/{paths['city']}"
        api.upload_folder(folder_path=src, path_in_repo=f"{f}/{paths['city']}", repo_id=repo, repo_type="dataset",
                          ignore_patterns=["*.png", "*.geojsonseq", "*.pbf"])
        print("uploaded", src)


def pack_outputs(paths, name, files, skip=(".pt", ".safetensors", ".bin")):
    """Zip one notebook's new output files into <data>/share/<name>.zip, so they can be downloaded and
    shared in one go. `files` may hold files or folders; missing ones are skipped. Model weights are left out."""
    import zipfile
    share = f"{paths['root']}/share"
    os.makedirs(share, exist_ok=True)
    out = f"{share}/{name}.zip"
    added = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            todo = [f] if os.path.isfile(f) else [os.path.join(d, x) for d, _, xs in os.walk(f) for x in xs] if os.path.isdir(f) else []
            for x in todo:
                if x.endswith(skip):
                    continue
                arc = os.path.relpath(x, paths["root"]) if x.startswith(paths["root"]) else os.path.join("repo", os.path.basename(x))
                z.write(x, arc)
                added.append((arc, os.path.getsize(x)))
    for arc, size in added:
        print(f"  {arc:60s} {size / 1e6:8.1f} MB")
    print(f"\nPacked {len(added)} files into {out} ({os.path.getsize(out) / 1e6:.1f} MB).")
    print("Download it from Drive: My Drive > Navigate > data > share")
    return out
