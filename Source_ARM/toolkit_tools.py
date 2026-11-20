"""Inspect or test the USB toolkit. Self-tests target loopback only."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import xml.etree.ElementTree as ET

from usb_core import ROOT, atomic_json, now

VERSIONS = {
    'nmap': ['--version'], 'ncat': ['--version'], 'nc': ['-h'], 'ip': ['-V'],
    'ss': ['-V'], 'ping': ['-V'], 'curl': ['--version'], 'dig': ['-v'],
    'host': ['-V'], 'nslookup': ['-version'], 'busybox': [],
    'hashcat': ['--version'], 'clinfo': ['--version'], 'john': [],
    'hydra': ['-h'], 'crunch': ['-h'], 'ldapsearch': ['-VV'],
    'smbclient': ['--version'], 'rpcclient': ['--version'], 'nbtscan': ['-v', '-h'],
    'snmpwalk': ['--version'], 'onesixtyone': [], 'gobuster': ['--version'],
    'ffuf': ['-V'], 'lsof': ['-v'], 'jq': ['--version'], 'file': ['--version'],
    'strings': ['--version'], 'readelf': ['--version'], 'objdump': ['--version'],
    'capsh': ['--print'], 'capstone-job': ['list']}

# Several traditional command-line tools print help/version successfully but use
# status 1 because no operational target was supplied.
ALLOWED_CODES = {'nc': {0, 1}, 'hydra': {0, 255}, 'nbtscan': {0, 1, 2},
                 'onesixtyone': {0, 1}}


def execute(name, arguments, input_text=None, timeout=30):
    return subprocess.run([str(ROOT / 'toolkit/bin' / name), *arguments],
                          input=input_text, text=True, capture_output=True, timeout=timeout)


def status():
    records = {}
    for name, arguments in VERSIONS.items():
        result = execute(name, arguments)
        lines = (result.stdout + result.stderr).strip().splitlines()
        if result.returncode not in ALLOWED_CODES.get(name, {0}):
            raise RuntimeError(name + ' failed: ' + '\n'.join(lines)[:1000])
        detail = lines[0] if lines else 'ready'
        records[name] = detail
        print(name + ': ' + detail, flush=True)
    print('YOLO runs as the current user; raw packet operations still require OS privileges.')
    return records


def self_test():
    records = status()
    marker = b'CapStone toolkit loopback check\n'
    class LocalHTTP(BaseHTTPRequestHandler):
        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Nmap may deliberately close a probe before a response.
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(marker)))
            self.end_headers()
            self.wfile.write(marker)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), LocalHTTP)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = str(server.server_port)
    try:
        result = execute('nmap', ['-sT', '-sV', '--version-light', '-Pn', '-n', '-p', port,
                                  '--host-timeout', '30s', '127.0.0.1', '-oX', '-'], timeout=40)
        if result.returncode != 0:
            raise RuntimeError('Nmap loopback test failed: ' + result.stderr)
        tree = ET.fromstring(result.stdout)
        observed = tree.find("host/ports/port[@portid='" + port + "']")
        if observed is None or observed.find('state').get('state') != 'open':
            raise RuntimeError('Nmap did not observe the actual open loopback port')
        service = observed.find('service')
        if service is None or service.get('name') != 'http':
            raise RuntimeError('Nmap did not identify the local HTTP service')
        records['nmap_loopback'] = {'host': '127.0.0.1', 'port': int(port), 'state': 'open',
                                     'service': service.attrib}
        request = 'GET / HTTP/1.0\r\nHost: localhost\r\n\r\n'
        for name, arguments in (('nc', ['-w', '2', '127.0.0.1', port]),
                                ('ncat', ['--idle-timeout', '2s', '127.0.0.1', port])):
            result = execute(name, arguments, request, timeout=5)
            if result.returncode != 0 or marker.decode().strip() not in result.stdout:
                raise RuntimeError(name + ' did not read the actual HTTP response')
            records[name + '_loopback'] = 'HTTP response received'
        result = execute('curl', ['--noproxy', '*', '-fsS', 'http://127.0.0.1:' + port + '/'])
        if result.returncode != 0 or result.stdout != marker.decode():
            raise RuntimeError('Bundled Curl loopback request failed: ' + result.stderr)
        records['curl_loopback'] = 'HTTP response verified'
        result = execute('ip', ['-j', 'address', 'show'])
        if result.returncode != 0 or not isinstance(json.loads(result.stdout), list):
            raise RuntimeError('Bundled ip could not read local interface information')
        records['ip_local'] = 'JSON interface information parsed'
        result = execute('clinfo', ['-l'], timeout=60)
        if result.returncode != 0 or 'Platform' not in result.stdout or 'Device' not in result.stdout:
            raise RuntimeError('Bundled CPU OpenCL runtime was not detected: ' + result.stderr)
        records['opencl_cpu'] = result.stdout.strip().splitlines()
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory(prefix='capstone-hashcat-') as folder:
            folder = Path(folder)
            password = 'capstone-loopback-test'
            digest = hashlib.sha256(password.encode()).hexdigest()
            (folder / 'hash.txt').write_text(digest + '\n')
            (folder / 'words.txt').write_text('incorrect\n' + password + '\n')
            result = execute('hashcat', ['-m', '1400', '-a', '0', '--quiet', '--potfile-disable',
                                        '--outfile', str(folder / 'found.txt'), str(folder / 'hash.txt'),
                                        str(folder / 'words.txt')], timeout=180)
            if result.returncode not in (0, 1) or not (folder / 'found.txt').is_file() or password not in (folder / 'found.txt').read_text():
                raise RuntimeError('Hashcat CPU self-test failed: ' + (result.stdout + result.stderr)[-2000:])
        records['hashcat_cpu'] = 'Known synthetic SHA-256 recovered from a two-entry local list'
        result = execute('jq', ['-n', '-c', '{portable:true}'])
        if result.returncode != 0 or json.loads(result.stdout) != {'portable': True}:
            raise RuntimeError('Bundled jq functional check failed: ' + result.stderr)
        records['jq'] = 'Generated and parsed local JSON'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    receipt = {'date': now(), 'targets': ['127.0.0.1'], 'passed': True, 'checks': records}
    atomic_json(ROOT / 'logs/toolkit-self-test.json', receipt)
    print('Toolkit self-test passed; network checks targeted only a local loopback service.', flush=True)
    return receipt


def menu():
    import sys
    while True:
        print('\nUSB toolkit\n1) Show installed tools\n2) Test tools against loopback only\n3) Repair from cached Ubuntu packages\nEnter) Back')
        choice = input('Choose: ').strip()
        if not choice:
            return
        if choice == '1':
            status()
        elif choice == '2':
            self_test()
        elif choice == '3':
            subprocess.run([sys.executable, '-s', '-B', str(ROOT / 'setup_toolkit.py'), '--offline'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--menu', action='store_true')
    args = parser.parse_args()
    menu() if args.menu else self_test() if args.self_test else status()
