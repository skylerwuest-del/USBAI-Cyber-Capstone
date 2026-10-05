#!/usr/bin/env python3
"""Install a portable CAI environment on an exFAT USB, with offline wheel cache."""
import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

from capstone import ROOT, atomic_json, instance_lock
from setup import download, sha256

CAI = ROOT / "cai"
PYTHON = CAI / "python" / "bin" / "python3.12"
PACKAGES = CAI / "site-packages"
LOCK = ROOT / "cai-requirements.lock"


def install_python(asset):
    archive = CAI / "cache" / "python.tar.gz"
    download(asset, archive)
    if PYTHON.is_file():
        return
    with tempfile.TemporaryDirectory(prefix="capstone-python-") as temp:
        base = Path(temp)
        with tarfile.open(archive, "r:gz") as tar:
            # Validate entries and link targets before extracting on a filesystem
            # that supports links. Copying below materializes them for exFAT.
            for member in tar.getmembers():
                target = (base / member.name).resolve()
                if not target.is_relative_to(base) or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError("Unsafe portable Python archive entry")
                if member.issym() or member.islnk():
                    link = (target.parent / member.linkname).resolve() if member.issym() else (base / member.linkname).resolve()
                    if not link.is_relative_to(base):
                        raise ValueError("Portable Python link escapes archive")
            # Python 3.10 may not provide the extraction filter backport.
            import inspect
            kwargs = {"filter": "data"} if "filter" in inspect.signature(tar.extractall).parameters else {}
            tar.extractall(base, **kwargs)
        shutil.copytree(base / "python", CAI / "python", symlinks=False, dirs_exist_ok=True)
    PYTHON.chmod(0o755)


def pip_env():
    env = dict(os.environ)
    # Never inherit another project's virtualenv, PYTHONPATH, or index settings.
    for key in list(env):
        if key.startswith("PIP_") or key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
               LITELLM_LOCAL_MODEL_COST_MAP="True", DO_NOT_TRACK="1", OPENAI_AGENTS_DISABLE_TRACING="1",
               PYTHONDONTWRITEBYTECODE="1", TIKTOKEN_CACHE_DIR=str(CAI / "cache" / "tiktoken"))
    return env


def pip(*args):
    subprocess.run([str(PYTHON), "-m", "pip", *args], cwd=ROOT, env=pip_env(), check=True)


def make_lock(wheels):
    lines = []
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            package = BytesParser().parsebytes(archive.read(metadata))
        lines.append(package["Name"] + "==" + package["Version"] + " --hash=sha256:" + sha256(wheel))
    if not lines:
        raise RuntimeError("No dependency wheels were downloaded")
    LOCK.write_text("# Ubuntu x86_64 / portable CPython 3.12; all wheels hash-pinned.\n" + "\n".join(sorted(lines)) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Reinstall using the USB's existing archive, wheels, and lock")
    args = parser.parse_args(argv)
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("The CAI bundle targets x86_64 Ubuntu")
    for directory in (CAI / "cache", CAI / "wheels", ROOT / "state"):
        directory.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "cai-assets.json").read_text())
    with instance_lock(ROOT / "state" / "runner.lock"):
        if args.offline:
            archive = CAI / "cache" / "python.tar.gz"
            if not archive.exists() or sha256(archive) != assets["python"]["sha256"]:
                raise RuntimeError("Verified Python archive missing for offline setup")
        install_python(assets["python"])
        wheels = CAI / "wheels"
        if not LOCK.exists():
            if args.offline:
                raise RuntimeError("Offline setup needs cai-requirements.lock and cached wheels")
            download(assets["cai"], wheels / assets["cai"]["file"])
            # Resolve only binary wheels usable on Ubuntu 22.04's glibc 2.35 or older.
            platforms = []
            for minor in range(35, 16, -1):
                platforms += ["--platform", "manylinux_2_" + str(minor) + "_x86_64"]
            for tag in ("manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"):
                platforms += ["--platform", tag]
            pip("download", "--dest", str(wheels), "--index-url", "https://pypi.org/simple",
                "--only-binary=:all:", "--python-version", "3.12", "--implementation", "cp",
                "--abi", "cp312", "--abi", "abi3", "--abi", "none", *platforms,
                "-r", str(ROOT / "cai-core-requirements.txt"))
            make_lock(wheels)
        # Install into a fresh sibling so a failed repair leaves the old bundle usable.
        pending = CAI / "site-packages.pending"
        if pending.exists():
            shutil.rmtree(pending)
        pip("install", "--no-index", "--find-links", str(wheels), "--require-hashes",
            "--no-deps", "--no-compile", "--target", str(pending), "-r", str(LOCK))
        env = pip_env()
        env["PYTHONPATH"] = str(pending)
        subprocess.run([str(PYTHON), "-c", "import cai, openai, litellm; import importlib.metadata as m; assert m.version('cai-framework') == '0.5.10'; print('CAI import check passed')"], env=env, check=True)
        if not args.offline:
            subprocess.run([str(PYTHON), "-c", "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"], env=env, check=True)
        previous = CAI / "site-packages.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if PACKAGES.exists():
            PACKAGES.rename(previous)
        pending.rename(PACKAGES)
        atomic_json(CAI / "installed.json", {"cai_version": assets["cai"]["version"],
                     "python_version": "3.12", "lock_sha256": sha256(LOCK)})
        if previous.exists():
            shutil.rmtree(previous)
    print("CAI installed on USB. Launch with: bash launch.sh --cai", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("CAI setup failed: " + str(exc))
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nCAI setup interrupted; rerun bash setup-cai.sh.")
        raise SystemExit(130)
