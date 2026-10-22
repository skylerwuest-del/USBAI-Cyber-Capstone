# Startup menu and first-prompt revision — 2026-10-20

Startup now contains only model choices 1–3 and exit. Each successful choice opens the
original application menu without starting a server or chat. That menu's chat/queue/serve
actions receive the selected session-only GGUF model; choosing a saved model explicitly
replaces the override. Original options 1–10 remain and optional benchmarking is now 11.
AirLLM commands do not inherit a GGUF override. Saved `config.json` and model weights
are unchanged. Chat still uses the existing real Server/Sessions lifecycle and telemetry.

Interactive GGUF prompts now advertise terminal/process tools directly with concise
app-owned descriptions and disable SDK tool-discovery bridge overhead. Actual function
names, argument schemas, required fields and constraints are retained, with no vendored
SDK edits. Queue/review tool definitions and AirLLM remain as before. Existing tiny-model
profiles also apply to flat drop-in copies, and Coder 7B has the bundled Qwen tool template.
Qwen 9B has narrow experimental suitability evidence from the user's successful file task;
this is not a general tool-reliability certification.

Logical batches and physical microbatches are both 512, reducing counter granularity
from the old 2048-token logical default while retaining the original 512 microbatch size.
Progress polls every five seconds, describes first-batch evaluation instead of showing a
misleading zero, and handles the installed runtime's list-shaped `next_token` field.
It distinguishes prompt counts from generation counts. Old performance measurements are
not reused under the new prompt/batch identity. CPU evaluation is still real computation;
menu changes alone do not make a weak model correctly execute tool calls.

Bundled-Python regression suite: **129 tests, 128 passed**, one existing AirLLM test
fails because optional PyTorch is absent. Tests cover selected-model propagation, chat
return behavior, explicit saved-model switching, AirLLM isolation, preserved JSON schemas,
small batch settings, honest zero/batch progress, and dict/list generation counters.
The current user's active chat was left running; real integration checks use a temporary
source copy and separate loopback port, shared weights opened for inference, and a harmless `/tmp`
file task. No user desktop file or task queue was changed.

The final real Qwen 9B test created and read back a temporary text file via real
Hermes terminal calls, returned to the original menu, then to startup, and exited.
First prompt evaluation measured **1247 tokens in 42.99 seconds**,
compared with the prior user log of 3653 tokens in 94.82 seconds. These are
different file-task prompts, not a controlled benchmark; the first-prompt workload
is substantially smaller. The console showed the first completed 512-token count
at 19 seconds. The full two-tool creation/readback conversation took 78.03 seconds.
No progress-thread exception occurred in the final test. CPU response time remains
material; no instant-response guarantee is made. Measured details append to the
primary diagnostics text file under MENU FLOW AND REAL TOOL VALIDATION.

---

# Unified setup validation — 2026-10-20

`bash setup.sh` now performs online preparation; `bash setup.sh --offline` repairs
from verified caches. Real online preparation and a subsequent offline repair both passed
on this Ubuntu 24.04 x86_64/glibc 2.39 host. No sudo or system Python installation was used.
Existing installed GGUF files were retained. The pinned Python/Hermes archives and 63
hash-verified Python wheel packages are cached locally. The absent legacy Ubuntu-library
archive was replaced by a separately documented host-library cache with its own SHA-256
manifest, rather than bypassing verification of the original asset.

The real llama.cpp server loaded an installed SmolLM2 model, passed readiness and model
identity checks, and the real Hermes SDK reached `You>`. The setup smoke check then quit.
A separate `launch.sh --model ... --chat` check sent a fixed harmless greeting, received a
real `Hermes>` reply, and exited cleanly. No autonomous tasks or password audits ran.
The setup receipt is `state/setup-ready.json`; detailed preparation output is in
`logs/setup.log`, and structured checks append to `logs/llm-fit-diagnostics.txt`.
The runtime's timestamped request timings were used to add a sampler-format regression
check, including use of the server-reported decode speed.

Automatic menu option 1 was also exercised through the actual `launch.sh`: it selected
Qwen2.5 Coder 7B, loaded it, reached the real Hermes prompt in 8.6 seconds, then returned
to the startup menu and exited. Final bundled-Python suite: **120 tests, 119 passed, one
failed** because the separate existing AirLLM cache test requires uninstalled PyTorch.
All eight setup tests and the real-runtime timing-format regression test passed.

Eight setup tests cover library cache creation/offline reuse, archive safety, missing/corrupt
offline assets, installed-model preservation, the unified command sequence, readiness only
after successful verification, and pinned wheel downloads before offline installation.
Earlier missing-runtime statements below describe the pre-setup state and are historical.
AirLLM/PyTorch and the optional toolkit were not installed or altered by this GGUF setup.
Compatibility of the host-generated library cache with other OS releases is not implied.

---

# Passive performance tracking and explicit benchmark validation — 2026-10-20

Added numeric-only passive timing collection for interactive startup selections, same-host
measurement reuse as a bounded ranking adjustment, and explicit startup option 5 /
`bash launch.sh --benchmark-installed`. The original RAM/context/runtime gates run before
history is considered. Default and manual selection still bypass ranking. No benchmark
runs at normal startup, and no model weights or AirLLM components are changed.

Latest combined suite: **111 tests, 110 passed, one failed** (the existing AirLLM
cache test requires absent PyTorch). All 27 new performance tests and all 42 fit tests
passed. A startup/session fixture verifies that loader/warmup timing lines are excluded
and only later request timings are reused. Actual shell menu, `--status`, and explicit
benchmark missing-runtime behavior were checked; syntax and ignore-rule checks passed.

New synthetic tests cover signature invalidation (hardware/model/runtime/context/threads),
fresh available-RAM rejection despite favorable history, minimum sample thresholds,
expiry, bounded scoring/cache size, preserved capability scoring, failure penalties,
corrupt/unwritable cache handling, passive numeric log parsing, no raw text retention,
partial lines, no double counting, explicit benchmark dispatch, identical harmless
requests, response timings, cleanup, and RAM-pressure skip behavior.

The local runtime/SDK/PyTorch limitations described below still apply. All performance
values in unit tests are synthetic fixtures; no real-model speed or reliability has been
measured or claimed. Full test outcomes append to the primary diagnostics text file.

---

# Offline LLM Fit Phase 1 validation — 2026-10-20

This section describes checks on this downloaded folder. The older validation notes below
are historical and do not establish the presence of a working runtime in this copy.

Implemented startup selection uses the existing Server, Sessions and cleanup paths.
Unit tests cover RAM pressure/reserves, CPU/runtime ABI, GPU presence vs support, bounded
GGUF parsing/discovery, native context, profile evidence, deterministic scoring, session-only
overrides, no confirmation, default/manual paths, bounded fallback, readiness/model identity,
diagnostic appends, offline scan behavior, CLI overrides/check/status/queue delegation,
AirLLM configuration and independent durable-job state. A synthetic local HTTP server
exercises real server-process startup, health checks, `/props` and cleanup without inference.
Existing queue, review, console, download and AirLLM protocol tests also run.

Latest validation results: **42/42 new fit tests passed**, including the real local
synthetic server test. The combined regression suite has **84 tests: 83 passed, one
failed** because the existing AirLLM weight-cache test imports unavailable PyTorch.
No skips were used to hide the missing dependency. Initial sandbox-only runs also
recorded socket-permission errors; rerunning with localhost access resolved those.

All development test outcomes (including environment failures) are appended under
`[AUTOMATED TEST RUN]` in `logs/llm-fit-diagnostics.txt` by `run_fit_tests.py`.
The runner captures verbose output and returns nonzero on failure. No suite runs during
normal launch. Socket-based fixture tests need permission to bind localhost.

Local limitations: `hermes/python/bin/python3.12`, the Hermes SDK and `runtime/llama-server`
are absent. System Python 3.12 is used for tests. The existing AirLLM CPU cache test requires
PyTorch, which is absent; no dependencies were installed or repaired. Real GGUF metadata
was read from all four installed models; no weights were modified and no real tasks were
executed. Actual inference, tokens/sec, peak RSS and real Hermes tool reliability are
**not verified** here. Interactive prompt sequencing is tested with session fixtures;
actual SDK readiness is logged only when its worker reaches that point.

Manual shell checks verify startup menu, exit, missing-runtime return-to-menu and
`bash launch.sh --status`. Configured default and original AirLLM files are unchanged.
The weights remain under `models/drop-in/`. Full acceptance on a prepared USB still needs
a real server/SDK launch: choose auto/default/manual, enter chat, `/quit`, repeat, and
verify saved config is unchanged and diagnostics report matching model paths/readiness.
No real inference or task-queue run should be inferred from synthetic tests.

---

# CapStone Hermes validation — 2026-10-05

The installed USB program uses the real Hermes Agent SDK from release 0.21.5,
tag `v2026.9.24`, commit `f97608f178d1ffeca59860195ab7da295f7c8e5f`.
The old CAI program is backed up under
`/home/charles/Work/CyberCapstone-backup-2026-10-05`.

## Real autonomous model run

Used the actual Qwen3.5-9B Q4_K_M weights and CPU llama.cpp server, with 65,536-token
context. No scripted model responses were used for this test.

The task asked Hermes to inspect a deliberately broken `add(a, b)` implementation,
fix subtraction to addition, write unittest coverage for positive, negative and zero
inputs, run it and fix failures. Recorded tool calls included:

- `read_file` to inspect the original code.
- `patch` to change `return a - b` to `return a + b`.
- `write_file` to create tests.
- `terminal` to run unittest: **3 test methods, 9 assertions, passing**.
- A fresh review session re-read both files and ran the tests independently.

The first review returned prose instead of completion JSON. The runner refused to
mark the task complete and started another pass. After adding the structured verdict
step, the real model produced a valid positive review with recorded observations,
the task was marked complete and the owned model process was stopped. The model's
prose misstated the assertion count; the actual test file contains 9 assertions.

Evidence is stored under `logs/validation-hermes/`, including actual requests, tool
results and `autonomous-edit-and-test.json`. That first run used an isolated test
workspace under `/home/charles/Work/CapStone-rebuild`, with model weights served from
the USB. It did not modify the user's existing work files.

Observed host performance: approximately 10.7 GB model resident RAM, around 4 generated
tokens/second and about 41 prompt tokens/second. This was a CPU-only run on the host's
Intel i9-11900H. These are observations, not a speed guarantee for another computer.

## Actual Ubuntu userland + USB run

Downloaded the official `ubuntu:26.04` linux/amd64 OCI filesystem and verified its layer
SHA-256. `/etc/os-release` identifies **Ubuntu 26.04.1 LTS**. Image manifest digest:
`sha256:88a381d5b5eeb2b35d3ad70925a362c37ce569daf43ede89ff818ec20e4d3794`.
The receipt is `logs/validation-hermes/ubuntu-image.json`.

Bubblewrap isolated this Ubuntu userland, mounted the actual exFAT USB at `/mnt`, and
ran the actual `bash launch.sh --query ...` entry point. The model and every Hermes
tool action ran inside that Ubuntu filesystem. The real model autonomously created
`workspace/.validation-2026-10-05/ubuntu-proof.txt` and read it back. The file contains
exactly `Hermes executed this task on Ubuntu` plus a newline, 36 bytes.

The fresh reviewer listed the directory and read the actual file. It attempted an
unknown completion tool once; Hermes rejected that call, and the model then returned
the required valid completion JSON. Successful actual observations, rather than that
rejected tool call, were used for completion. The queue finished with exit status 0;
the launcher closed the worker and model, then flushed the USB filesystem.

Progress: `state/validation-ubuntu.json`. Real session receipts: `logs/sessions/`.
The user's `tasks.txt` was not executed for this test; a separate validation request
and progress record were used.

The Ubuntu image initially lacked `libgomp.so.1`. Official Ubuntu libgomp, libstdc++ and
libgcc packages were obtained using APT's signed metadata and package verification.
Their libraries are bundled as regular files in `runtime/`; package SHA-256 receipts
are in `runtime/ubuntu-libraries.json`. The model binary then ran in the clean Ubuntu
image. `state/ubuntu-runtime-libs.tar.gz` and `assets.json` support verified offline
restoration of these libraries.

An actual terminal session also ran `launch.sh --chat` inside Ubuntu, displayed the
Hermes prompt, accepted `/quit`, and exited 0 with model shutdown and USB flush.

This validates Ubuntu's userland on the existing host kernel and CPU. A separately
booted Ubuntu desktop or a different target machine has not been physically tested.

## Regression and preservation checks

The 11 task-loop tests passed using the USB's portable Python. They cover plain-English
queues without VERIFY, empty/unsupported completion claims, failed observations,
interrupted reviews, incomplete evidence, fresh-session retries, completed-task resume,
changed requests, session crashes and cleanup of background commands after timeout.

The terminal-editor workflow was exercised with a real editor subprocess in a temporary
folder: it saved a backup and accepted a task containing no VERIFY command. The actual
USB menu and `--check`/`--status` entry points passed smoke checks.

`setup_hermes.py --offline` successfully reinstalled all 63 hash-pinned binary dependency
wheels and imported the real Hermes SDK inside Ubuntu using the USB's cached archives.
No system Python packages were installed.

Two additional download tests passed inside Ubuntu, for 13 tests total: resuming without
host curl and correctly restarting when a server ignores Range instead of corrupting
the final file. Both verify the checksum and reuse a valid cached file without network
requests. The real Hugging Face model-menu metadata lookup also passed inside clean
Ubuntu using the USB's certificate bundle. The initial lookup failed because the clean
image lacked system CA certificates; the bundle now supplies its trusted CA file.

Compared every original workspace file byte for byte against the backup and compared
all original task prompts and VERIFY lines: **preserved**. Only obsolete task-file
comment instructions were updated. Both the original 4.68 GB model and new 5.68 GB model
remain on the drive. No symlinks exist in the Hermes, AirLLM or model runtime bundles.

## Original AirLLM experiment (before the Hermes integration)

AirLLM 4.0.0, Transformers 5.8.0 and CPU-only PyTorch 2.6.0 are installed separately.
Actual CPU layer splitting and token generation passed with a tiny locally generated
synthetic Llama model, both on the host and inside Ubuntu 26.04.1. This uses real
AirLLM streaming code, with no GPU or CUDA runtime.

At the time of the original rebuild, this proved the portable CPU experiment worked with that tiny model. It does not prove
useful language quality, acceptable huge-model speed, compatibility with every model
architecture, or an AirLLM-backed Hermes agent. At that stage, the task harness used
GGUF through llama.cpp. The integration below supersedes that limitation. Large Transformers weights and layer copies must fit the USB;
the experiment checks space before downloading them.


## AirLLM + Hermes integration

Menu 9 now launches the actual Hermes SDK against a CPU AirLLM service, with either
interactive terminal tools or the same persistent worker/reviewer task loop. Its submenu
includes Hugging Face Transformers repository downloads, installed-model selection,
task editing, queue progress and dependency repair. The default GGUF path is preserved.

A real public Qwen3-1.7B checkpoint is installed on the USB, pinned to revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its original weights were verified against
published SHA-256 values and split by AirLLM into reusable USB-local layer files.
Inference and all validation tool actions used the Ubuntu 26.04.1 userland described
above, mounted at a different path from the host's USB mount. No cloud inference or
scripted model responses were used in the real-model sessions.

Actual Hermes tool receipts include mkdir, shell file writes, cat and directory
inspection. This establishes that model outputs are converted into real executed
commands, rather than just displayed as instructions. Early diagnostic runs also
exposed the small model's poor exact-newline handling, confused reviewer behavior,
invalid review JSON and incomplete inspection. Those runs were not marked successful
by the developer or used to claim a correct exact-content result. The later basic
file-action test also saw a fresh reviewer repeat the write before reading the file,
despite its inspection-only role. Review roles are prompted, not an OS-enforced
read-only sandbox; this checkpoint is therefore not a reliable completion judge. Their progress
records are retained separately from the user's queue.

The final adapter includes a compact AirLLM role prompt, lower process-local SDK
context floor without claiming a fictional context capacity, checked token budgets,
forced initial reviewer tool selection, strict review-field validation, rejection of
malformed/truncated tool calls, CPU garbage-collection throttling, a bounded CPU weight
cache, and cancellation when a client disconnects. These changes do not make a weak
model reliable at every task or make AI review a proof of correctness.

All **30 regression/integration tests passed inside Ubuntu**, using the USB's Python
3.12 and CPU PyTorch. The test log is `logs/validation-airllm-tests.txt`. Tests include
native-tool conversion, literal delimiters inside shell arguments, tool-choice and
schema validation, streaming response framing, real HTTP busy/health behavior,
disconnected-client cancellation, bounded CPU caching with correct tensor retention,
queue retry/resume semantics and background-command timeout cleanup.
Both default and AirLLM `--check` commands passed inside Ubuntu. The actual main menu
opened submenu 9 and returned/exited correctly. Ctrl+C diagnostic shutdown closed its
owned model and worker and flushed USB writes.

Before this integration, launcher/configuration/task/prompt files were backed up to
`/home/charles/Work/CapStone-before-airllm-20261005-153549`. The user's `tasks.txt`,
`prompt.txt` and default `config.json` were compared byte for byte against that backup
and remain unchanged. Validation tasks use separate folders under `workspace/` and
separate state files. The user's task queue was not executed during validation.


### Completed real AirLLM task-loop run

`bash launch.sh --backend airllm --query ... --state state/validation-airllm-cache.json`
created and read `workspace/.validation-airllm-cache/proof.txt` through actual Hermes
terminal calls. The observed bytes are `b'AirLLM and Hermes executed this task\n'`.
The task requested a basic sentence write/read, rather than the earlier exact-newline
case. The review returned correctly typed positive JSON with recorded file-read
observations. The queue recorded completion on pass 1, the command exited 0, and the
launcher reported process closure and a USB filesystem flush. The review also repeated
the write before reading, as disclosed above; this validates lifecycle/tool integration
and does not establish a reliable inspection-only reviewer with this checkpoint.

Real worker receipt: `logs/sessions/1791240835260686073-work`.
Real reviewer receipt: `logs/sessions/1791241107545125939-review`.
Task state: `state/validation-airllm-cache.json`.
The final cached CPU run generated at about one token per second after its prompt pass.
The server logged a 7.68 GB cache budget on the 32 GB host. Speed and available-memory
budgets vary by machine and model.

The real Ubuntu terminal menu also passed: option 9 → 1 started AirLLM and displayed
the Hermes prompt; `/quit` closed the model/agent and flushed USB writes, returned to
the AirLLM menu, then Back → 0 exited the main menu with status 0.


## Follow-up console, small-model and toolkit validation — 2026-10-06

The user's report concerned main menu 1 (regular GGUF/Hermes), not AirLLM. This launcher
uses the actual Hermes SDK with a simple console; the absence of the full Hermes TUI
was intentional. The SDK's quiet mode could start a raw animation on terminal output.
The menu also ran its supervisor in the foreground process group, so Ctrl+C could reach
both processes before the menu forwarded an additional interrupt. That could interrupt
cleanup itself. The USB wrapper now disables the SDK animation, owns a line-based status
thread, starts the supervisor in its own session, forwards one interrupt and shields
worker/model cleanup from repeated interrupts. A StopRequested exception escapes the
SDK's ordinary retry/KeyboardInterrupt handling. Vendor SDK files were not edited.

The regular 9B model remains selected. CPU defaults now use a compact system prompt,
16,384-token context, zero hidden-thinking budget and a smaller chat output limit.
Native GGUF context metadata caps every model's configured context; process-local SDK
compatibility no longer forces a fictitious 64K window for small GGUFs. Autonomous task
sessions retain their separate, larger output budgets. Greetings use a chat-specific
brief and do not trigger assessment tasks.

The actual Qwen2.5 0.5B GGUF answered `hello?` through Hermes and exited normally with
`/quit`. The actual 9B GGUF also answered `hello?`, acknowledging its red-team role,
after roughly 90 seconds of cold CPU prompt processing/generation. Those are observations
on this host, not speed promises for another Ubuntu computer.

Added hash-verified, revision-pinned SmolLM2-360M-Instruct Q8_0 (386,404,992 bytes) and
Qwen2.5-0.5B-Instruct Q4_K_M (491,400,032 bytes). Qwen 0.5B Transformers weights are
also installed for AirLLM. The model source receipts are next to the weight files;
`small-model-assets.json` records the GGUF pins. Small models are communication/latency
smoke-test choices, not reliable agents or independent reviewers. Existing larger weights
were preserved. The user's tasks.txt was not executed or changed.

The standing `prompt.txt` now focuses on authorized red-team assessment, connected-network
and service discovery, evidence, controlled validation, findings and remediation. It
preserves unrelated files and responds normally to greetings. Original files, tasks,
workspace, state and logs were backed up to
`/home/charles/Work/CapStone-before-followup-20261006` before changes.

The restricted tool sandbox exposed a read-only view of the USB, initially mistaken
for a host mount problem. Host-side verification confirmed the actual mount is writable.
The kernel had reported an unclean exFAT unmount, so the standard UDisks filesystem
Check was run while temporarily unmounted and reported consistent=true. The drive was
remounted read-write; no formatting or filesystem repair was performed.

All **36 tests passed inside Ubuntu 26.04.1**, including new real-signal tests showing
repeated interrupts cannot abort shutdown and the original handlers are restored even
when cleanup raises. Status-thread tests show late callbacks cannot restart output after
closure. GGUF metadata tests cover variable-length metadata, truncation and native context.
The log is `logs/validation-followup-tests.txt`.

The portable toolkit uses Nmap/Ncat 7.98, OpenBSD Netcat, iproute2, ping, Curl, Bind DNS
utilities and BusyBox. All 94 cached dependency packages were downloaded without installation
and checked against SHA-256 values in signed Ubuntu repository metadata. The toolkit
contains its private dynamic loader, shared libraries, Nmap data, trusted CA file and
package copyrights, all as regular files for exFAT. It changes no host packages or root
configuration, and does not grant capabilities/root. Its manifest is `toolkit/manifest.json`.

The actual USB `setup_toolkit.py --offline` rebuilt the tools from cached archives inside
clean Ubuntu, extracting into temporary /tmp space and materializing symlinks. The actual
USB toolkit self-test passed there: Nmap TCP-connect scanning and version detection observed
the open port and HTTP service of a temporary loopback server, Netcat/Ncat read its real
HTTP response, Curl verified its body and ip returned valid interface JSON. Only 127.0.0.1
was scanned. The receipt is `logs/toolkit-self-test.json`. No connected LAN assessment or
user task was run during development.


A PTY owned inside the Ubuntu userland exercised the actual **main menu 1**, started
a real 9B request, and sent repeated terminal Ctrl+C. The launcher closed/flushed,
returned to the main menu, emitted no further model status during an observation window,
left no listener on 8080 or worker process, and exited 0 when menu 0 was selected.
The transcript is `logs/validation-menu-interrupt.txt`. This uses the same foreground
terminal signal path as an ordinary Ubuntu terminal, without sending Ctrl+C to the
outer container-test wrapper.

## Portable assessment toolkit expansion — 2026-10-07

Downloaded 257 Ubuntu 26.04 amd64 packages and their dependency closure without installing
them on the host. Every package was matched by filename and SHA-256 against the already
downloaded, signed Ubuntu repository metadata. The resulting cache is about 203 MB; the
expanded toolkit is about 1.9 GB including repair caches and wordlist data.

The official Kali Linux `wordlists_2026.2.0_all.deb` package was downloaded from Kali's
package pool and pinned as SHA-256
`6b08a6df43dd81d75c74d8010865bbf75cec00286baa9919088466d22a800377`.
Its `rockyou.txt.gz` was expanded to `toolkit/share/wordlists/rockyou.txt`: 139,921,507
bytes, SHA-256
`16035fea7742cb0561c513de1d946eda5716d7de294e6c732449740096686173`.

Hashcat 7.1.2 loaded the bundled PoCL CPU device and recovered the known password
`capstone-loopback-test` from a two-entry local list against a synthetic SHA-256 digest.
John the Ripper 1.9.0 separately recovered `capstone-john-test` from a two-entry list
against a synthetic SHA-512 crypt hash inside the clean Ubuntu userland. These tests used
only generated local test credentials.

Version/startup checks passed for Hashcat, Clinfo, Hydra, Crunch, LDAP clients, Samba
clients, NetBIOS/SNMP tools, Gobuster, FFUF, lsof, jq, file, binutils and Linux capability
inspection tools through the private loader. The setup script rebuilt the bundle offline
from its caches, and Python syntax checks passed. The outer development sandbox blocks
network sockets, so the existing Nmap/Netcat/Curl loopback suite was not repeated during
this expansion; its earlier actual Ubuntu result remains documented above.


Additional actual Ubuntu greeting checks passed through Hermes: SmolLM2 360M GGUF
answered in 1.94 seconds and Qwen2.5 0.5B AirLLM answered in 15.83 seconds, each followed
by `/quit` and exit 0 with model shutdown/flush. Startup and first layer splitting are
excluded from those greeting durations. The receipt is `logs/validation-small-chats.json`.
The earlier Qwen 0.5B GGUF greeting recorded 3.72 seconds of prompt evaluation plus 0.19
seconds of generation in the actual model server log.


The actual Qwen 0.5B GGUF Hermes session then called its terminal tool with
`nmap --version`. It found the USB-packaged Nmap through the normal tool PATH,
returned actual version 7.98 with exit code 0, and the model reported that version.
The tool receipt is `logs/sessions/1791323006303500273-work/result.tools.json`.
The session exited 0 with `/quit`. This establishes that the toolkit is exposed to
Hermes itself, rather than only callable by a standalone diagnostic script.

## Long-running task-loop repair — 2026-10-08

Inspection of the user's password-audit run found 29 worker passes. Several session
directories contained tool events but no `result.json`; their start times were about 2,406
seconds apart and their state notes recorded exit 124. This matched the configured
2,400-second fixed worker deadline. Other receipts showed John being terminated after a
300-second tool timeout, followed by attempts to detach password-cracking commands. Worker
cleanup then removed tracked processes before the next fresh pass could use them.

The fixed worker deadline is now disabled. Foreground terminal commands and process waits
may use the configured 604,800-second limit, 32-turn worker segments automatically continue
inside the same agent process with the same conversation history, task ID and process
registry, and up to 16 such segments are permitted. Long-job prompts require the worker to
wait for the same bounded process instead of duplicating it. Review instructions reject
partial runs, timeouts, unsupported claims and abandoned background jobs as completion
evidence.

The existing audit was incorrectly marked complete after only the SHA-512 crypt account was
recovered; four yescrypt entries had not completed a RockYou pass. Its completion record was
preserved in the backup and reset to pending on the USB so the corrected loop can reassess it.
Fifteen task-loop unit tests pass, including new checks for disabled fixed deadlines,
max-iteration continuation classification and long-job completion requirements.

RockYou contains 14,344,393 lines. The user's actual four-salt yescrypt run reported about
59.8 candidate passwords per second, implying roughly 66.6 hours for a complete pass at that
rate. The earlier 12-hour value was therefore replaced with the seven-day command limit.
Actual time varies with CPU load, hash cost and the number of remaining salts.
