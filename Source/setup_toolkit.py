"""Rebuild the USB toolkit offline from checksum-pinned Ubuntu packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import gzip

from usb_core import ROOT, atomic_json, instance_lock

PROGRAMS = {name: 'usr/bin/' + name for name in (
    'nmap', 'ncat', 'ip', 'ss', 'ping', 'curl', 'dig', 'host', 'nslookup', 'busybox',
    'hashcat', 'clinfo', 'hydra', 'crunch', 'ldapsearch', 'ldapwhoami', 'smbclient',
    'rpcclient', 'nbtscan', 'snmpwalk', 'snmpget', 'snmpbulkwalk', 'onesixtyone',
    'gobuster', 'ffuf', 'lsof', 'jq', 'file', 'pw-inspector', 'strings', 'readelf',
    'objdump')}
PROGRAMS.update(nc='usr/bin/nc.openbsd', netcat='usr/bin/nc.openbsd',
                john='usr/sbin/john', getcap='usr/sbin/getcap',
                getpcaps='usr/sbin/getpcaps', capsh='usr/sbin/capsh')


def wrapper(name, binary):
    extras = {
        'nmap': 'set -- --datadir "$CAPSTONE_KIT/share/nmap" "$@"\n',
        'smbclient': 'set -- --configfile="$CAPSTONE_KIT/share/samba/smb.conf" "$@"\n',
        'rpcclient': 'set -- --configfile="$CAPSTONE_KIT/share/samba/smb.conf" "$@"\n',
    }
    extra = extras.get(name, '')
    if name == 'john':
        extra += '''CAPSTONE_ACCOUNT_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
if [[ -n "$CAPSTONE_ACCOUNT_HOME" && ! -f "$CAPSTONE_ACCOUNT_HOME/.john/john.conf" ]]; then
    mkdir -p "$CAPSTONE_ACCOUNT_HOME/.john"
    cp "$CAPSTONE_KIT/share/john/john.conf" "$CAPSTONE_ACCOUNT_HOME/.john/john.conf"
fi
export JOHN="$CAPSTONE_KIT/share/john"
'''
    return '''#!/bin/bash
set -euo pipefail
CAPSTONE_KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NMAPDIR="$CAPSTONE_KIT/share/nmap"
export OPENSSL_MODULES="$CAPSTONE_KIT/lib/ossl-modules"
export SSL_CERT_FILE="$CAPSTONE_KIT/share/cacert.pem"
export CURL_CA_BUNDLE="$CAPSTONE_KIT/share/cacert.pem"
export OCL_ICD_VENDORS="$CAPSTONE_KIT/etc/OpenCL/vendors"
export MIBDIRS="$CAPSTONE_KIT/share/snmp/mibs"
export SNMPCONFPATH="$CAPSTONE_KIT/etc/snmp"
export MAGIC="$CAPSTONE_KIT/share/misc/magic.mgc"
export HOME="$CAPSTONE_KIT/../state/tool-home"
export POCL_CACHE_DIR="$HOME/.cache/pocl"
mkdir -p "$HOME" "$POCL_CACHE_DIR"
''' + extra + 'exec "$CAPSTONE_KIT/lib/ld-linux-x86-64.so.2" --library-path "$CAPSTONE_KIT/lib/x86_64-linux-gnu:$CAPSTONE_KIT/lib/x86_64-linux-gnu/pocl:$CAPSTONE_KIT/lib:$CAPSTONE_KIT/lib/samba:$CAPSTONE_KIT/lib/pocl" "$CAPSTONE_KIT/libexec/' + name + '.real" "$@"\n'


def assemble(sysroot, destination):
    for folder in ('bin', 'etc', 'lib', 'libexec', 'share', 'licenses'):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    libroot = sysroot / 'usr/lib/x86_64-linux-gnu'
    # Flatten the dynamic-loader search directory, including BLAS aliases whose
    # system-wide alternatives links are normally created by package scripts.
    candidates = list(libroot.glob('*.so*')) + list((libroot / 'blas').glob('*.so*'))
    for source in candidates:
        if not source.is_file():
            continue
        target = destination / 'lib' / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError('Conflicting bundled library: ' + source.name)
        shutil.copy2(source, target, follow_symlinks=True)
    for name in ('ossl-modules', 'gconv'):
        if (libroot / name).is_dir():
            shutil.copytree(libroot / name, destination / 'lib' / name, symlinks=False, dirs_exist_ok=True)
    for name in ('samba', 'pocl'):
        if (libroot / name).is_dir():
            shutil.copytree(libroot / name, destination / 'lib' / name, symlinks=False, dirs_exist_ok=True)
    portable_arch_lib = destination / 'lib/x86_64-linux-gnu'
    portable_arch_lib.mkdir(parents=True, exist_ok=True)
    shutil.copy2(libroot / 'libpocl.so.2.13.0', portable_arch_lib / 'libpocl.so.2.13.0')
    shutil.copy2(libroot / 'libpocl.so.2.13.0', portable_arch_lib / 'libpocl.so.2')
    shutil.copytree(libroot / 'pocl', portable_arch_lib / 'pocl',
                    symlinks=False, dirs_exist_ok=True)
    # Hashcat derives its data directory from its executable location. Recreate
    # the distro's /usr/lib/hashcat and /usr/share/hashcat contents at the
    # portable base it calculates, resolving the distro symlinks as real files.
    shutil.copytree(sysroot / 'usr/lib/hashcat', destination / 'lib',
                    symlinks=False, dirs_exist_ok=True)
    for source in (sysroot / 'usr/share/hashcat').iterdir():
        if source.name in ('modules', 'bridges'):
            continue
        target = destination / 'lib' / source.name
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, target, follow_symlinks=True)
    # Hashcat dlopens the unversioned ICD loader name. Ubuntu's runtime package
    # only provides the versioned aliases, so preserve an exFAT-safe regular copy.
    shutil.copy2(libroot / 'libOpenCL.so.1.0.0', destination / 'lib/libOpenCL.so')
    for name, binary in PROGRAMS.items():
        source = sysroot / binary
        if not source.is_file():
            raise RuntimeError('Missing toolkit executable: ' + binary)
        shutil.copy2(source, destination / 'libexec' / (name + '.real'), follow_symlinks=True)
        target = destination / 'bin' / name
        target.write_text(wrapper(name, binary))
        target.chmod(0o755)
    job_wrapper = destination / 'bin/capstone-job'
    job_wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_JOB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$CAPSTONE_JOB_ROOT/hermes/python/bin/python3.12" -s -B "$CAPSTONE_JOB_ROOT/long_job.py" "$@"
''')
    job_wrapper.chmod(0o755)
    for source_name, target_name in (
            ('usr/share/nmap', 'share/nmap'), ('usr/share/hashcat', 'share/hashcat'),
            ('usr/share/john', 'share/john'), ('usr/share/pocl', 'share/pocl'),
            ('usr/share/snmp', 'share/snmp'), ('etc/snmp', 'etc/snmp'),
            ('usr/share/samba', 'share/samba'), ('etc/OpenCL', 'etc/OpenCL')):
        source = sysroot / source_name
        if source.is_dir():
            shutil.copytree(source, destination / target_name, symlinks=False, dirs_exist_ok=True)
    shutil.copy2(sysroot / 'etc/john/john.conf', destination / 'share/john/john.conf')
    (destination / 'share/misc').mkdir(parents=True, exist_ok=True)
    shutil.copy2(sysroot / 'usr/lib/file/magic.mgc', destination / 'share/misc/magic.mgc')
    rockyou_gzip = sysroot / 'usr/share/wordlists/rockyou.txt.gz'
    if not rockyou_gzip.is_file():
        raise RuntimeError('The verified Kali wordlists package did not contain rockyou.txt.gz')
    (destination / 'share/wordlists').mkdir(parents=True, exist_ok=True)
    with gzip.open(rockyou_gzip, 'rb') as source, (destination / 'share/wordlists/rockyou.txt').open('wb') as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    (destination / 'share/wordlists/README.txt').write_text(
        'rockyou.txt was extracted from the official Kali Linux wordlists package 2026.2.0.\n'
        'Use password lists only for systems and accounts you are authorized to assess.\n')
    certificates = ROOT / 'hermes/site-packages/certifi/cacert.pem'
    shutil.copy2(certificates, destination / 'share/cacert.pem')
    for doc in (sysroot / 'usr/share/doc').iterdir():
        copyright_file = doc / 'copyright'
        if copyright_file.is_file():
            shutil.copy2(copyright_file, destination / 'licenses' / (doc.name + '.copyright'))
    (destination / 'env.sh').write_text('''# Source this file to use the toolkit in your own terminal.
CAPSTONE_TOOLKIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CAPSTONE_TOOLKIT
export PATH="$CAPSTONE_TOOLKIT/bin:$PATH"
''')
    if any(path.is_symlink() for path in destination.rglob('*')):
        raise RuntimeError('The exFAT bundle must contain regular files, not symlinks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use cached .deb packages only (the default)')
    parser.parse_args()
    toolkit = ROOT / 'toolkit'
    manifest = json.loads((toolkit / 'manifest.json').read_text())
    if not shutil.which('dpkg-deb'):
        raise RuntimeError('Offline repair needs Ubuntu dpkg-deb; normal toolkit use needs no package manager')
    with instance_lock(ROOT / 'state/runner.lock'), tempfile.TemporaryDirectory(prefix='capstone-toolkit-') as folder:
        sysroot = Path(folder) / 'root'
        sysroot.mkdir()
        for package in manifest['packages']:
            source = toolkit / 'cache' / package['file']
            with source.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != package['sha256']:
                raise RuntimeError('Cached package checksum mismatch: ' + source.name)
            subprocess.run(['dpkg-deb', '-x', str(source), str(sysroot)], check=True)
        extra_manifest = json.loads((toolkit / 'extra-manifest.json').read_text())
        for package in extra_manifest['packages']:
            source = toolkit / 'cache-extra' / package['file']
            with source.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != package['sha256']:
                raise RuntimeError('Cached extra package checksum mismatch: ' + source.name)
            subprocess.run(['dpkg-deb', '-x', str(source), str(sysroot)], check=True)
        assemble(sysroot, toolkit)
        atomic_json(toolkit / 'files.json', {
            str(p.relative_to(toolkit)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder_name in ('bin', 'etc', 'lib', 'libexec', 'share', 'licenses')
            for p in (toolkit / folder_name).rglob('*') if p.is_file()})
        subprocess.run(['sync', '-f', str(ROOT)], check=True)
        print('USB toolkit rebuilt from verified cached Ubuntu packages and Kali wordlist data; host packages unchanged.')


if __name__ == '__main__':
    main()
