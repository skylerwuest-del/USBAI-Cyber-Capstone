#!/usr/bin/env python3
"""Preparation only: verify the local binary's real CLI/schema and record it."""
import json
from pathlib import Path
import sys

from select_model import ROOT, architecture, isolated_run, parse_hardware, validate_elf


def main():
    binary = ROOT / 'llmfit/linux' / architecture() / 'llmfit'
    validate_elf(binary, architecture())
    version = isolated_run(binary, ['--version']).strip()
    if not version.startswith('llmfit '):
        raise ValueError('Unexpected llmfit version output')
    help_text = isolated_run(binary, ['--help'])
    if '--no-dashboard' not in help_text or '--json' not in help_text:
        raise ValueError('Binary lacks required offline CLI options')
    raw = isolated_run(binary, ['--no-dashboard', 'system', '--json'])
    parse_hardware(json.loads(raw))
    (binary.parent / 'system.verified.json').write_text(raw)
    (binary.parent / 'version.txt').write_text(version + '\n')
    print('Verified ' + version + ' for ' + architecture())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit('llmfit preparation verification failed: ' + str(exc))
