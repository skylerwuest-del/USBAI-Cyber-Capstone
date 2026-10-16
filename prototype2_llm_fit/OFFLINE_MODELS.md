# Offline model selection

## Drag-and-drop models

Put complete `.gguf` files in `models/drop-in/`, finish copying, then launch
normally. Startup refreshes `models/models.json` automatically, entirely offline.
No background watcher or host installation is needed. To refresh separately, run
`python3 -B scripts/import_models.py` from this prototype directory.

The importer reads GGUF metadata/tensor dimensions and computes SHA-256 itself.
It supports dense Llama/Qwen2/Qwen3, selects 4096 context, estimates base memory
as file GiB × 1.25 + 1, and budgets KV bytes/token as layers × embedding × 16
(FP32 full-width K+V plus a 2× margin). These are unmeasured estimates. Larger
parameter counts receive higher generated priorities; hardware fit still gates
selection. Quantization is recorded as the GGUF `general.file_type` identifier.
Hybrid/recurrent, MoE, split, unsupported or malformed files are skipped with a
diagnostic; those require reviewed manual entries. Existing Qwen3.5 manual entries
remain supported by the selector. Automatic import does not establish publisher
authenticity or replace preparation-time backend compatibility testing.

Manually maintained entries/settings are preserved. Deleting an imported file
removes its generated entry on the next refresh. Manifest writes are locked and
atomic; malformed manifests are never overwritten. Models are never moved or
downloaded. Auto-import is the explicit opt-in for files placed in this folder;
other model files remain subject to the manual allowlist.

Run `bash linux.sh` from the repository root. It delegates to this prototype's
Linux entry, automatically selects a model, starts the existing llama.cpp server,
and starts Hermes. `--check` validates configuration without inference. The older
`launch.sh` menu remains a development/preparation entry and is not the strict
offline entry. The historical CAI prototype is unchanged.

## Selection contract

`scripts/select_model.py` uses only the Python standard library. It identifies
x86_64 or aarch64/arm64, checks the bundled llmfit ELF machine type, verifies its
version against `version.txt`, and runs `--no-dashboard system --json` with a
20-second timeout. Hardware JSON must contain `system.total_ram_gb` and
`system.available_ram_gb`; both are GiB despite the upstream names. Other hardware
fields are retained for diagnostics, including CPU, GPUs, VRAM, backend, and unified
memory. Unknown schemas, missing libraries, invalid binaries, failures and timeouts
produce a diagnostic and CPU-only `/proc/meminfo` fallback. Failure to read reliable
available memory stops selection. Kernel RAM readings cap llmfit's rounded values.

Only `models/models.json` entries qualify. Files must resolve inside this project's
`models/` directory, have a GGUF v2/v3 header with bounded metadata/tensor tables,
aligned tensor offsets within the file, readable metadata and
native context, and match the preparation-approved SHA-256. Missing or invalid entries
are logged and excluded. An explicit `--model` request must pass the same checks;
it never substitutes another model. Removing a JSON entry removes eligibility.

The custom algorithm computes:

```
budget_GiB = max(0, min(total_RAM, available_RAM) - 2) * 0.85
base_GiB = max(estimated_ram_gb, file_bytes / 2**30 * 1.10 + 0.5)
required_GiB = base_GiB + context_size * kv_bytes_per_token / 2**30
```

MemAvailable already excludes occupied OS memory. The additional 2 GiB reserves
future Ubuntu, agent and server growth, and the 15% margin leaves headroom.
Highest numeric priority wins **only among fitting models**; ties prefer lower
requirements then name. Context remains the manifest value and cannot exceed native
context (minimum 4096 for this agent). Threads use half the logical CPU count.
The server uses one slot and bounded batching. RAM is checked again before restarts.
Runtime selection does not rewrite `config.json`.

GPU offload is deliberately zero: the reported backend/VRAM does not prove that
the bundled llama.cpp supports it or that VRAM is currently free. Discrete VRAM
is never added to RAM; unified RAM is counted once. No throughput estimate is
invented. GPU offload/performance optimization remains unsupported.

## Why use a separate fit algorithm?

Upstream's [actual system JSON emitter](https://github.com/AlexsJones/llmfit/blob/main/llmfit-tui/src/main.rs)
was inspected, rather than the abbreviated fields in CLI help. Its
[hardware implementation](https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/hardware.rs)
uses binary GiB. Upstream supports local `custom_models.json` and
`LLMFIT_CUSTOM_MODELS`, as documented in
[custom models](https://github.com/AlexsJones/llmfit/blob/main/docs/custom-models.md).
Those entries supplement its catalog; they do not authenticate our files or map
recommendations to GGUF paths. This integration therefore invokes only hardware
profiling, never catalog recommendations, downloads, update, benchmark, or sharing.
Memory, KV allowance, priority and launch settings are local decisions.

No llmfit binary is currently supplied or claimed verified. During preparation,
`scripts/verify_llmfit.py` executes the actual installed CLI under the same network
filter, checks the schema, and records its version and output. This must succeed
on each architecture before enabling llmfit on a prepared USB.

## USB preparation (separate development machine, online allowed)

1. Prepare the existing Hermes source, portable Python 3.12 and hash-pinned binary
   dependencies with the existing setup workflow. For each architecture bundle
   `hermes/linux/{x86_64,arm64}/python/bin/python3.12` and matching `site-packages/`.
   Shared Hermes source remains `hermes/src`. The legacy `hermes/python` layout is
   accepted when compatible. Neither current prototype has a `/tmp` venv rebuilding
   workflow; this change does not add a package-manager step to startup.
2. Obtain an audited upstream llmfit release, record its tag/commit and checksum,
   and bundle executable `llmfit/linux/{x86_64,arm64}/llmfit`. Include its MIT license.
   If prebuilt binaries do not meet your Ubuntu baseline, build the pinned source
   on a separate matching development machine using
   `cargo build --locked --release -p llmfit`, after checking the chosen release's
   Cargo workspace/package name. Copy its `target/release/llmfit`; never build on
   the target host. Cross compilation needs the corresponding Linux target and
   linker; native builds on each architecture are preferable.
3. Inspect `file`, `readelf -h`, `readelf -l`, and `readelf -d` for each executable.
   Verify its interpreter, minimum glibc version and every required shared library
   on the oldest supported Ubuntu image. A release archive is not proof of static
   linking. Prefer compatible static llmfit builds; bundle permitted runtime
   libraries with the application when needed. Do the same for portable Python,
   llama.cpp and its adjacent ggml/llama libraries. Bundle CPU-compatible
   `runtime/linux/{x86_64,arm64}/llama-server`; the legacy runtime path is accepted
   only after an ELF architecture check. Validate CPU ISA baseline on real targets.
4. On each matching architecture run
   `python3 -B scripts/verify_llmfit.py`. Keep generated `version.txt` on the USB;
   review `system.verified.json` before release. An unverified binary falls back
   rather than being trusted.
5. Copy approved GGUF files to `models/`, independently validate them with the
   bundled llama.cpp, and record SHA-256 in `models/models.json`. Entries require
   `name`, project-relative `file` (for example `models/example.gguf`),
   `parameters_b`, `quantization`, integer `context_size`, `estimated_ram_gb`,
   `kv_bytes_per_token`, integer `priority`, and lowercase `sha256`.
   Memory fields use GiB/bytes. Set KV bytes from the actual architecture/cache
   types, including recurrent state where applicable; measure peak memory at the
   selected context/batching and increase estimates when necessary. Existing
   manifest estimates are unverified allowances, not measured guarantees.
6. Test with the network disconnected. Never invoke setup or the interactive
   download menus during normal offline use. No model/binary assets are downloaded
   by selection or Linux startup.

## Network and host persistence

llmfit runs in a disposable `/tmp` home/cache with a sanitized environment and a
Linux seccomp filter inherited by subprocesses. Socket and socketpair creation
are denied, including alternate-ABI bypasses. This prevents provider probes as
well as Internet traffic; seccomp failure prevents llmfit execution and triggers
fallback. The main application uses an unprivileged user/network namespace with
only loopback enabled so Hermes can reach llama-server. It fails closed if the
host disables user namespaces; it does not change global firewall rules or use sudo.
Offline environment flags also disable common telemetry/update paths.
State, logs and agent configuration stay on the USB; helper temporary files stay
under `/tmp`. Tool commands inherit the offline namespace, so network security
tools cannot reach external targets through this entry.

## Automated and integration validation

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests-offline -v`.
Fixtures require neither model weights nor GPU. Real seccomp tests verify kernel
enforcement across exec; mocked sockets alone are not proof of isolation.

Before USB release, use clean Ubuntu x86_64 and ARM64 VMs with NICs detached:

1. Set RAM to 4/12/32 GiB and supply small/medium/large approved local models.
   Run `bash linux.sh --check` and compare selection to the documented formula.
2. Run `bash linux.sh --serve-only`, confirm loopback health, then start normal
   Hermes chat and exercise a local file tool. Repeat after moving the USB mount.
3. Remove llmfit, corrupt its version file, and substitute a wrong-architecture
   ELF. Confirm fallback diagnostics and safe selection. Repeat with all models
   missing, malformed manifest and symlink escape; expect clean failure.
4. Repeat with a NIC attached and an external capture plus `strace -f -e trace=network`.
   Confirm no external packets; llmfit socket calls must fail EPERM and application
   requests stay on loopback in its namespace. Compare host home/package trees
   before/after; only USB state and `/tmp` runtime files may change.
5. Exercise namespace-disabled and seccomp-disabled policies; verify documented
   failure/fallback, no installation prompts, and no network-capable llmfit exec.
6. Measure peak RAM at full context and sustained tool use; adjust manifest
   estimates. Estimates cannot guarantee against simultaneous host memory pressure,
   incorrect preparation metadata, or model/backend incompatibility.

These VM/GPU/inference checks require prepared assets and are not unit-test claims.

## Implementation validation in this checkout

- 21 offline selector tests passed, including low/mid/high RAM fixtures, invalid
  files/manifests, traversal/symlinks, architecture checks, fallback, exact model
  requests, command validation and seccomp/descendant checks.
- Existing task-loop tests: 18 passed. Existing console/GGUF tests: 6 passed.
- `bash -n` for both Linux entries and `git diff --check` passed.
- The full existing Hermes suite was attempted: 37 tests ran, with four errors
  from missing PyTorch and sandbox-denied local HTTP servers (including a class
  setup error). An attempted run outside the sandbox was interrupted; it is not
  claimed passed. No dependencies were installed to bypass these limitations.
- `bash linux.sh --check` exited cleanly with the missing ARM64 portable Python
  preparation diagnostic. No full inference or installed llmfit CLI validation
  was possible because binaries/models/dependencies are absent.

Created files: root/prototype `linux.sh`, `scripts/select_model.py`,
`scripts/gguf_check.py`, `scripts/verify_llmfit.py`, `scripts/offline_launch.py`,
`models/models.json`, `tests-offline/test_selection.py`, and this document.
Updated integration files: `capstone.py`, `usb_core.py`, `agent_worker.py`,
`task_loop.py`, root `README.md`, and `.gitignore`. Existing launcher menus and
historical CAI files were retained. No new Python dependency was introduced.

Limitations: CPU-only selection; manually validated memory/KV estimates;
unprivileged namespaces required; real ARM64/x86_64 USB integration still needs
testing. Each selection hashes eligible files, which costs USB read time. The
structural validator and approved hash do not replace preparation-time llama.cpp
validation of tensor types, model architecture and numerical contents. Run on
ordinary Ubuntu hosts/VMs; memory-constrained containers require separately
accounting for their cgroup limits before deployment.
