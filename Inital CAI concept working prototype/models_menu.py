#!/usr/bin/env python3
"""Download public GGUF models from Hugging Face and select a USB-local model."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from capstone import ROOT, atomic_json, instance_lock
from setup import download

SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def parse_url(url):
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username or parsed.password:
        raise ValueError("Use an https://huggingface.co/owner/repository URL")
    parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/")]
    if len(parts) < 2 or any(p in ("", ".", "..") or "/" in p for p in parts[:2]):
        raise ValueError("Missing or invalid repository owner/name")
    repo = "/".join(parts[:2])
    if len(parts) == 2:
        return repo, "main", ""
    if len(parts) >= 4 and parts[2] == "tree":
        return repo, parts[3], ""
    if len(parts) >= 5 and parts[2] in ("blob", "resolve"):
        filename = "/".join(parts[4:])
        if not filename.lower().endswith(".gguf") or any(p in (".", "..", "") for p in parts[4:]):
            raise ValueError("The file URL must point to a .gguf model")
        return repo, parts[3], filename
    raise ValueError("Use a repository URL or a direct GGUF blob/resolve URL")


def metadata(repo, revision):
    endpoint = "https://huggingface.co/api/models/" + urllib.parse.quote(repo, safe="/")
    endpoint += "/revision/" + urllib.parse.quote(revision, safe="") + "?blobs=true"
    with urllib.request.urlopen(endpoint, timeout=30) as response:
        return json.load(response)


def select(items, title):
    print("\n" + title)
    for i, item in enumerate(items, 1):
        print(str(i) + ") " + item)
    value = input("Number (Enter cancels): ").strip()
    if not value:
        return None
    index = int(value) - 1
    if not 0 <= index < len(items):
        raise ValueError("Choose one of the listed numbers")
    return index


def shard_group(filename, files):
    match = SPLIT.fullmatch(filename)
    if not match:
        return [filename]
    prefix, _, total_text = match.groups()
    total = int(total_text)
    expected = [prefix + "-" + str(i).zfill(5) + "-of-" + total_text + ".gguf" for i in range(1, total + 1)]
    if any(name not in files for name in expected):
        raise ValueError("Incomplete split GGUF in repository")
    return expected


def selectable_names(names):
    result = []
    for name in names:
        match = SPLIT.fullmatch(name)
        if match is None or match.group(2) == "00001":
            result.append(name)
    return sorted(result)


def set_model(path):
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("This file does not have a GGUF header")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["model"] = str(path.relative_to(ROOT))
    atomic_json(ROOT / "config.json", cfg)
    print("Selected model: " + cfg["model"])


def pull(url=None):
    url = url or input("Hugging Face repository or GGUF URL: ").strip()
    repo, revision, filename = parse_url(url)
    print("Looking up public GGUF files in " + repo + "...", flush=True)
    info = metadata(repo, revision)
    files = {f["rfilename"]: f for f in info["siblings"] if f["rfilename"].lower().endswith(".gguf")}
    if not files:
        raise ValueError("No GGUF files found. Choose a GGUF quantization repository.")
    if not filename:
        names = selectable_names(files)
        labels = [name + " (" + format(files[name].get("size", 0) / 1e9, ".2f") + " GB; first shard if split)" for name in names]
        choice = select(labels, "Available GGUF models")
        if choice is None:
            return
        filename = names[choice]
    if filename not in files:
        raise ValueError("GGUF file not found at this repository revision")
    group = shard_group(filename, files)
    pinned = info["sha"]
    identifier = repo.replace("/", "--") + "--" + pinned[:12]
    dest = ROOT / "models" / identifier
    dest.mkdir(parents=True, exist_ok=True)
    required = sum(files[name].get("size", 0) for name in group)
    if shutil.disk_usage(ROOT).free < required + 100 * 1024 * 1024:
        raise RuntimeError("Not enough free USB space for this download")
    print("Downloading " + str(len(group)) + " file(s) onto USB. Revision: " + pinned, flush=True)
    receipts = []
    with instance_lock(ROOT / "state" / "runner.lock"):
        for name in group:
            f = files[name]
            lfs = f.get("lfs", {})
            digest = lfs.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Hugging Face did not provide a SHA-256 for " + name)
            # Preserve subdirectories and reject traversal supplied by repository metadata.
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid repository filename")
            asset = {"url": "https://huggingface.co/" + urllib.parse.quote(repo, safe="/") + "/resolve/" + pinned + "/" + urllib.parse.quote(name, safe="/"),
                     "sha256": digest, "size": f["size"]}
            download(asset, dest / relative)
            receipts.append(dict(asset, file=name))
        atomic_json(dest / "source.json", {"repository": repo, "revision": pinned, "files": receipts})
        set_model(dest / group[0])
    print("Model downloaded and selected. Its license is on the Hugging Face repository page.")


def choose_installed():
    paths = list((ROOT / "models").rglob("*.gguf"))
    relative = {str(p.relative_to(ROOT / "models")): p for p in paths}
    names = selectable_names(relative)
    if not names:
        print("No installed GGUF models. Use the download menu first.")
        return
    choice = select(names, "Installed USB models")
    if choice is not None:
        shard_group(names[choice], relative)
        with instance_lock(ROOT / "state" / "runner.lock"):
            set_model(relative[names[choice]])
