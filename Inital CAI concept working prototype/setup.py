#!/usr/bin/env python3
"""Download pinned Ubuntu CPU runtime and model onto this USB. No pip packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(asset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == asset["size"]:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        if sha256(path) == asset["sha256"]:
            print("Verified existing " + path.name, flush=True)
            return
    partial = path.with_name(path.name + ".part")
    if partial.exists() and partial.stat().st_size > asset["size"]:
        raise RuntimeError("Oversized partial file: " + str(partial))
    if not partial.exists() or partial.stat().st_size != asset["size"]:
        subprocess.run(["curl", "--fail", "--location", "--retry", "8", "--retry-delay", "3",
                        "--connect-timeout", "20", "--speed-time", "60", "--speed-limit", "1024",
                        "--continue-at", "-", "--output", str(partial), asset["url"]], check=True)
    # Flush the downloaded bytes before accepting a cached checksum as success.
    # USB removal after setup must not leave a verified file only in page cache.
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    print("Checking SHA-256 of " + partial.name, flush=True)
    if partial.stat().st_size != asset["size"] or sha256(partial) != asset["sha256"]:
        raise RuntimeError("Checksum mismatch: " + str(partial) + "; delete this partial file and rerun setup")
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def extract_runtime(archive):
    # exFAT does not support symbolic links. Materialize linked shared libraries as files.
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        servers = [m for m in members.values() if PurePosixPath(m.name).name == "llama-server" and m.isfile()]
        if len(servers) != 1:
            raise RuntimeError("Expected exactly one llama-server in runtime archive")
        directory = PurePosixPath(servers[0].name).parent
        def file_content(member, visited=None):
            visited = set() if visited is None else visited
            if member.name in visited:
                raise RuntimeError("Cyclic archive link")
            visited.add(member.name)
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                target = link if member.islnk() else PurePosixPath(member.name).parent / link
                # Normalize link references inside the in-memory archive index only.
                parts = []
                for part in target.parts:
                    if part == "..":
                        if not parts:
                            raise RuntimeError("Archive link escapes root")
                        parts.pop()
                    elif part not in (".", ""):
                        parts.append(part)
                name = "/".join(parts)
                linked = members.get(name) or members.get("./" + name)
                if linked is None:
                    raise RuntimeError("Missing archive link target: " + name)
                return file_content(linked, visited)
            if not member.isfile():
                raise RuntimeError("Unexpected runtime member type")
            return tar.extractfile(member)
        runtime = ROOT / "runtime"
        runtime.mkdir(exist_ok=True)
        count = 0
        for member in members.values():
            p = PurePosixPath(member.name)
            if p.parent != directory or not (p.name == "llama-server" or ".so" in p.name):
                continue
            dest = runtime / p.name
            with file_content(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            dest.chmod(0o755)
            count += 1
        print("Installed " + str(count) + " runtime files (exFAT compatible)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    if args.runtime_only and args.model_only:
        parser.error("Choose only one of --runtime-only and --model-only")
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("This runtime targets x86_64 Ubuntu; other architectures need a matching runtime")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required. On Ubuntu run: sudo apt install curl")
    manifest = json.loads((ROOT / "assets.json").read_text())
    (ROOT / "state").mkdir(exist_ok=True)
    if not args.model_only:
        archive = ROOT / "state" / "llama-ubuntu-x64.tar.gz"
        download(manifest["runtime"], archive)
        extract_runtime(archive)
    if not args.runtime_only:
        download(manifest["model"], ROOT / "models" / manifest["model"]["file"])
    (ROOT / "licenses").mkdir(exist_ok=True)
    sources = {
        "llama.cpp-LICENSE": "https://raw.githubusercontent.com/ggml-org/llama.cpp/" + manifest["runtime"]["version"] + "/LICENSE",
        "Qwen-LICENSE": "https://huggingface.co/" + manifest["model"]["repository"] + "/resolve/" + manifest["model"]["revision"] + "/LICENSE",
    }
    for name, url in sources.items():
        if not (ROOT / "licenses" / name).exists():
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(ROOT / "licenses" / name), url], check=True)
    print("Setup complete. Start with: bash launch.sh", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted; rerun bash setup.sh to resume.")
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("Setup failed: " + str(exc))
        raise SystemExit(1)
