#!/usr/bin/env python3
"""Download pinned Ubuntu CPU runtime and model onto this USB. No pip packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import ssl
import subprocess
import tarfile
import time
import urllib.request

from portable_env import activate

ROOT = Path(__file__).resolve().parent
activate(ROOT)


def tls_context():
    certificates = ROOT / 'hermes/site-packages/certifi/cacert.pem'
    return ssl.create_default_context(cafile=str(certificates) if certificates.exists() else None)


def python_download(asset, partial):
    """Resume without requiring curl on the Ubuntu host."""
    context = tls_context()
    for attempt in range(9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == asset['size']:
            return
        headers = {'User-Agent': 'CapStone-USB/2'}
        if offset:
            headers['Range'] = 'bytes=' + str(offset) + '-'
        try:
            request = urllib.request.Request(asset['url'], headers=headers)
            with urllib.request.urlopen(request, context=context, timeout=60) as response:
                status = response.status
                if status == 206:
                    if not response.headers.get('Content-Range', '').startswith('bytes ' + str(offset) + '-'):
                        raise RuntimeError('Server returned an unexpected byte range')
                elif status == 200:
                    # A server may ignore Range; restart rather than append a full body.
                    offset = 0
                else:
                    raise RuntimeError('Unexpected download HTTP status: ' + str(status))
                mode = 'ab' if offset else 'wb'
                report_at = time.monotonic() + 5
                with partial.open(mode) as target:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        if offset + len(block) > asset['size']:
                            raise RuntimeError('Download exceeds the published size')
                        target.write(block)
                        offset += len(block)
                        if time.monotonic() >= report_at:
                            print('Downloaded %.2f / %.2f GB' % (offset / 1e9, asset['size'] / 1e9), flush=True)
                            report_at = time.monotonic() + 5
            if offset != asset['size']:
                raise OSError('Download ended before the published size')
            return
        except OSError:
            if attempt == 8:
                raise
            print('Download interrupted; resuming...', flush=True)
            time.sleep(3)


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
        python_download(asset, partial)
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


def extract_runtime(archive, destination=None, libraries_only=False):
    # exFAT does not support symbolic links. Materialize linked shared libraries as files.
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        servers = [m for m in members.values() if PurePosixPath(m.name).name == "llama-server" and m.isfile()]
        if not libraries_only and len(servers) != 1:
            raise RuntimeError("Expected exactly one llama-server in runtime archive")
        directory = PurePosixPath(servers[0].name).parent if servers else None
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
        runtime = Path(destination) if destination else ROOT / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        count = 0
        for member in members.values():
            p = PurePosixPath(member.name)
            if (not libraries_only and p.parent != directory) or not (p.name == "llama-server" or ".so" in p.name):
                continue
            dest = runtime / p.name
            with file_content(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            dest.chmod(0o755)
            count += 1
        print("Installed " + str(count) + " runtime files (exFAT compatible)", flush=True)


def prepare_gpu_runtimes(offline=False):
    """All downloads, runtime libraries and extraction stay on the USB."""
    assets = json.loads((ROOT / 'gpu-assets.json').read_text())
    for backend, entries in assets.items():
        destination = ROOT / 'runtime' / backend
        for index, asset in enumerate(entries):
            archive = ROOT / 'state' / 'gpu-archives' / asset['file']
            ensure_asset(asset, archive, offline)
            extract_runtime(archive, destination=destination, libraries_only=index > 0)


def restore_ubuntu_libraries(asset):
    archive = ROOT / asset['file']
    if not archive.is_file() or archive.stat().st_size != asset['size'] or sha256(archive) != asset['sha256']:
        raise RuntimeError('Offline Ubuntu runtime archive is missing or corrupt: ' + str(archive))
    runtime = ROOT / 'runtime'
    runtime.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or Path(member.name).name != member.name or not member.name.startswith(('libgomp.so', 'libstdc++.so', 'libgcc_s.so', 'libssl.so', 'libcrypto.so')):
                raise ValueError('Unexpected file in Ubuntu runtime archive')
            dest = runtime / member.name
            with tar.extractfile(member) as source, dest.open('wb') as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
    print('Ubuntu runtime libraries restored from verified USB archive.', flush=True)


def ensure_asset(asset, path, offline=False):
    if offline:
        if not path.is_file() or path.stat().st_size != asset['size'] or sha256(path) != asset['sha256']:
            raise RuntimeError('Offline asset missing or corrupt: ' + str(path) + '; run bash setup.sh online first')
    else:
        download(asset, path)


def prepare_runtime_libraries(asset, offline=False):
    """Reuse original archives or cache the compatible host's installed libraries."""
    record = ROOT / 'state/ubuntu-runtime-libs.local.json'
    candidates = [asset]
    if record.exists():
        candidates.insert(0, json.loads(record.read_text()))
    for candidate in candidates:
        archive = ROOT / candidate['file']
        if archive.is_file() and archive.stat().st_size == candidate['size'] and sha256(archive) == candidate['sha256']:
            restore_ubuntu_libraries(candidate)
            return
    if offline:
        raise RuntimeError('Runtime library cache missing or corrupt; run bash setup.sh online first')
    runtime = ROOT / 'runtime'
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime))
    inspection = subprocess.run(['ldd', str(runtime / 'llama-server')], env=env,
                                capture_output=True, text=True, check=True)
    if 'not found' in inspection.stdout:
        raise RuntimeError('Host cannot supply required runtime libraries: ' + inspection.stdout.strip())
    required = {'libgomp.so.1', 'libstdc++.so.6', 'libgcc_s.so.1', 'libssl.so.3', 'libcrypto.so.3'}
    libraries = {}
    for line in inspection.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] in required and fields[1] == '=>':
            source = Path(fields[2]).resolve()
            if not source.is_file():
                raise RuntimeError('Runtime library not a regular file: ' + str(source))
            libraries[fields[0]] = source
    if not {'libstdc++.so.6', 'libgcc_s.so.1'}.issubset(libraries):
        raise RuntimeError('Could not identify the required host C++ runtime libraries')
    # Only cache a host bundle which can actually execute this verified server.
    subprocess.run([str(runtime / 'llama-server'), '--version'], env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, timeout=30)
    archive = ROOT / 'state/ubuntu-runtime-libs.tar.gz'
    partial = archive.with_suffix('.gz.part')
    with tarfile.open(partial, 'w:gz') as tar:
        for name, source in sorted(libraries.items()):
            tar.add(source, arcname=name, recursive=False)
    os.replace(partial, archive)
    from usb_core import atomic_json
    generated = dict(file='state/ubuntu-runtime-libs.tar.gz', size=archive.stat().st_size,
                     sha256=sha256(archive), provenance='Libraries supplied by this host OS; not the legacy archive',
                     host=platform.platform(), libraries={name: dict(source=str(path), sha256=sha256(path))
                                                           for name, path in libraries.items()})
    atomic_json(record, generated)
    restore_ubuntu_libraries(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.runtime_only and args.model_only:
        parser.error("Choose only one of --runtime-only and --model-only")
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("This runtime targets x86_64 Ubuntu; other architectures need a matching runtime")
    manifest = json.loads((ROOT / "assets.json").read_text())
    (ROOT / "state").mkdir(exist_ok=True)
    if not args.model_only:
        archive = ROOT / "state" / "llama-ubuntu-x64.tar.gz"
        ensure_asset(manifest["runtime"], archive, args.offline)
        extract_runtime(archive)
        prepare_gpu_runtimes(args.offline)
    if not args.runtime_only:
        ensure_asset(manifest["model"], ROOT / "models" / manifest["model"]["file"], args.offline)
    if not args.model_only and manifest.get('ubuntu_libraries'):
        prepare_runtime_libraries(manifest['ubuntu_libraries'], args.offline)
    (ROOT / "licenses").mkdir(exist_ok=True)
    sources = {
        "llama.cpp-LICENSE": "https://raw.githubusercontent.com/ggml-org/llama.cpp/" + manifest["runtime"]["version"] + "/LICENSE",
        "Qwen-LICENSE": "https://huggingface.co/" + manifest["model"]["repository"] + "/resolve/" + manifest["model"]["revision"] + "/LICENSE",
    }
    for name, url in sources.items():
        if not (ROOT / "licenses" / name).exists():
            if args.offline:
                raise RuntimeError('Missing offline license file: ' + name)
            with urllib.request.urlopen(url, context=tls_context(), timeout=60) as response:
                (ROOT / 'licenses' / name).write_bytes(response.read())
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
