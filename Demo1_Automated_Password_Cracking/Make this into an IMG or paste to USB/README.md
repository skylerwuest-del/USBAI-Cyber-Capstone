# CapStone USB — Hermes in the terminal

Start from the USB folder:

```bash
bash launch.sh
```

The default path serves a USB-local GGUF model with llama.cpp and connects **Hermes Agent
0.21.5** to it. Hermes has real terminal, process, file-reading, file-writing, patching,
search, and task-planning tools. It can execute commands and edit files itself. Everything
is displayed in the terminal. No webpage, browser UI, cloud account, or API subscription
is needed for the default workflow. A loopback HTTP API carries requests between Hermes
and the model process; that API is not a webpage. CAI and LiteLLM are no longer used.

The installed bundle targets **Ubuntu 26.04 LTS, x86_64**, with **32 GB RAM and no GPU**.
Python 3.12, Hermes source, its binary dependencies, model weights and server are on
the USB. Dependencies use regular files rather than a symlink-based virtual environment,
so the exFAT drive is supported. The drive must be mounted with permission to execute
its Python and model-server binaries. Launch with Bash because exFAT does not preserve
normal Unix executable bits.

## Menu

1. Interactive Hermes: starts the local model and an actual tool-using conversation.
2. Edit `tasks.txt`: edit the preordained queue using a terminal editor.
3. Run/resume tasks, then exit: keeps making attempts until each task passes review;
   closes Hermes, its tracked processes and the model server, flushes USB writes, and
   exits the launcher when the queue is complete.
4. Show progress: display `state/task-loop.json`.
5. Pull a GGUF from a Hugging Face URL: choose a quantization, download it to the USB,
   verify its published SHA-256 and select it. Split GGUFs are downloaded together.
6. Select an installed GGUF.
7. Edit `prompt.txt`: standing instructions included in worker/chat sessions.
8. Serve only: terminal-only model service; Ctrl+C shuts it down.
9. AirLLM + Hermes: interactive terminal/file tools or the autonomous task queue,
   with Transformers model download and selection.
10. USB toolkit: inspect installed tools, test a local loopback service, or rebuild
    from the cached Ubuntu packages.
0. Exit.

`CAPSTONE_EDITOR` or `EDITOR` selects the editor; otherwise nano, vim or vi is used.
Editing saves a `.bak` copy. Models, tasks and progress stay on the USB.

## Terminal console, YOLO and quick testing

This launcher uses the **real Hermes SDK** with its own simple terminal console, rather
than Hermes's full-screen TUI. The `You>` and `Hermes>` prompts are intentional. It displays
`YOLO ON`: `HERMES_YOLO_MODE=1` auto-approves supported tool actions. YOLO is a launch
mode of the standard Hermes build, not a separate version. Tools run as the logged-in
Ubuntu user. This does not supply a sudo password, grant root, or remove Hermes's built-in
hard restrictions.

The moon animation is disabled in the USB console. Owned line-based CPU progress stops
when the request/session ends. The menu starts the supervisor in a separate process
session and forwards one Ctrl+C; cleanup ignores repeated interrupts while it closes
workers, the model, and flushes the USB. Ctrl+C returns to the menu; `/quit` exits chat.
Clearing the terminal does not stop a process—use Ctrl+C or `/quit`.

The regular 9B model remains selected. Its hidden reasoning budget is now zero to avoid
spending hundreds of CPU-generated tokens before an ordinary answer. Model-specific
settings in `model-profiles.json` choose context, output budget and tools, and GGUF
metadata caps the server context to the model's advertised native capacity. Chat has a
shorter output budget than autonomous task sessions. The local compatibility setting
does not modify the vendor SDK or advertise an invented context size.

For quick feedback, choose **main menu 6** and select one of:

| Model | Installed GGUF | Purpose |
| --- | --- | --- |
| SmolLM2 360M Instruct | Q8_0, 386 MB | Very small conversation/launch smoke tests |
| Qwen2.5 0.5B Instruct | Q4_K_M, 491 MB | Fast conversation and basic tool experiments |

Both are pinned to repository revisions and verified against their published SHA-256
values. These small models are test models, not dependable red-team agents or completion
judges. The original 7B and 9B models are preserved. To switch back, use menu 6 again.
Qwen 0.5B is also installed as Transformers weights: **menu 9 → 4** selects it for AirLLM.
GGUF through llama.cpp is the faster default for small CPU models; AirLLM is useful for
exercising the layer-streaming backend.

`prompt.txt` is the standing context file, editable through menu 7. It now focuses on
**authorized cyber red-team assessments**, network/service discovery, observed evidence,
controlled validation, findings and remediation. It still instructs the agent to answer
greetings normally and preserve unrelated files.

## Plain-English tasks

Put your tasks in **`tasks.txt`**, separating them with a line containing `---`:

```text
Inspect calculator.py and fix addition. Add tests for positive, negative and zero inputs.
Run the tests and fix failures.
---
Write a short README explaining how to run calculator.py.
```

**No `VERIFY:` line is required.** Existing `VERIFY:` lines remain compatible as extra
checks, and old JSON queues still work. `tasks.example.txt` provides a new plain-English
example. Your original task prompts and optional checks were preserved during migration.

For unattended execution, starting the model and task loop together:

```bash
bash launch.sh --run --tasks tasks.txt
```

Or select menu option **3**. `startup.sh` is an optional shortcut for interactive Hermes;
it installs no Ubuntu login hook. To use it for the queue:

```bash
bash startup.sh --run --tasks tasks.txt
```

`launch.sh` is the normal entry point. `setup.sh` repairs the pinned model/server and
reinstalls the cached Hermes dependencies. `setup_hermes.py --offline` repairs Hermes
without network access. You do not need to run setup on every launch. The supplied
runtime, source archives, wheel cache and hash-pinned dependency lock support moving
the drive to another compatible Ubuntu computer. Model downloads need internet access;
running installed models and local tasks does not.
The model downloader falls back to portable Python when the host has no curl and uses
the USB's certificate bundle for HTTPS lookups.

## How the task loop decides to stop

The loop follows the Ralph idea while keeping active work intact. A worker uses 32-turn
segments; reaching that turn budget automatically continues with the same conversation,
Hermes process registry and tracked commands. It no longer has a fixed 40-minute wall-clock
cutoff. Up to 16 segments are available before the outer loop needs a fresh worker, and
**the outer task loop has no attempt limit**. Errors, exhausted workers, missing evidence,
and negative reviews cause another attempt. Restarting the launcher resumes unfinished
tasks. Changes to a task invalidate its earlier completion; unchanged completed tasks are
skipped.

Long terminal jobs use the USB-local `capstone-job` supervisor. It starts one detached,
named command, records its PID, command, timestamps, output and exit status under
`state/jobs/`, refuses accidental duplicate starts, and remains alive across fresh AI
turns. Workers inspect or wait for that same job instead of restarting it. Tools such as
John also use their own named recovery session on the USB. The Hermes batch timeout is
disabled for these waits while the normal seven-day terminal ceiling remains available.
Ctrl+C still stops the model/task runner; a durable job can be inspected or deliberately
stopped with `capstone-job status NAME` or `capstone-job stop NAME`.

The AI receives the queue task ID through `CAPSTONE_QUEUE_TASK_ID` and uses it as the
durable job name. `capstone-job list`, `status`, `log` and `wait` are read-only inspection
operations. `start` refuses a duplicate or an already-recorded result; after inspecting a
failed job and its tool-specific recovery file, `start --restart NAME -- COMMAND` archives
the old status before launching the recovery command.

A fresh reviewer uses tools to inspect the actual files and run appropriate checks against
the original request. The worker saying “done” never completes a task. A reviewer must
return a positive structured verdict with evidence and no remaining work, and the runner
must have recorded successful independent observations. If a model returns prose, a
structured local model call formats the review's actual observations into a
verdict. Optional legacy checks run afterward. Each task's original request remains the
acceptance criterion; there is no agent-editable completion marker.
llama.cpp constrains that JSON during generation; the AirLLM adapter requests JSON and
validates its fields afterward, rejecting invalid output for another attempt.
AirLLM reviews require an initial inspection tool call before allowing a final verdict.
Exact file-content requirements call for raw-byte checks, including newline counts.
For password audits and scans, a partial result, timeout, unsupported claim or abandoned
background process is insufficient; the reviewer is explicitly required to establish that
the requested targets or candidate set were actually processed. Full RockYou password
audits have an additional non-model validator: completion requires a zero-exit durable
John job whose provenance includes RockYou and `--format=crypt`, whose log shows hashes
were loaded, and whose log contains John's `Session completed` exhaustion marker. A
previously accepted audit lacking that evidence is reopened automatically.

This is AI review, not a mathematical proof. A model can miss requirements; task clarity
and the selected model affect results. An impossible task, unavailable privilege, failed
hardware, or missing input cannot be made complete by looping. Such tasks remain open
and retry rather than receiving a fabricated success. **Ctrl+C** interrupts the run and
saves progress. Power loss can interrupt an action after it has executed but before its
result is saved; fresh sessions are instructed to inspect before repeating actions.

Runtime records:

- `state/task-loop.json`: task status, attempts, progress, review evidence.
- `logs/sessions/`: original requests, real tool observations and final responses.
- `logs/server.log`: llama.cpp startup and inference output.
- `state/hermes/`: USB-local Hermes configuration and session data.
- `state/jobs/`: durable command status, history and output logs.
- `workspace/`: default folder for task files. Existing user files were retained.

To rerun an unchanged queue with a separate progress record:

```bash
bash launch.sh --run --tasks tasks.txt --state state/another-run.json
```

Hermes executes as the current Ubuntu user and has that user's terminal and filesystem
permissions. It cannot conjure a sudo password or bypass operating-system permissions.
The unattended configuration approves ordinary supported tool actions automatically;
Hermes's own hard restrictions still apply. Commands the agent deliberately detaches
outside its tracked tool processes, host services started through service managers, or
processes surviving forced power loss cannot be promised to close automatically.

## Models and CPU performance

The selected model is **Qwen3.5-9B Q4_K_M**, a 5.68 GB quantization by Unsloth of Qwen's
model. The old Qwen2.5-Coder-7B model is preserved. The new model has native support for
long context and tool use. The default Hermes/llama.cpp path now uses a compact system prompt and a
16,384-token context for CPU responsiveness. The local-only SDK compatibility setting
uses the actual server context rather than forcing a 64K floor. Both local backends have a process-local context
compatibility setting. Select models that actually support their configured
context and tool calling.
Smaller or poorly instruction-following models may print commands instead of calling
tools, or fail review repeatedly. Changing harnesses cannot guarantee model competence.

The observed 9B CPU test used roughly 10.7 GB resident RAM and generated about 4 tokens
per second on this host. First prompt processing took about two minutes. Tasks involving
many tool calls can take minutes. Your Ubuntu CPU and USB speed will affect timings.
`config.json` controls the selected model, port, CPU threads, context, generation length,
session limits, command timeouts and retry delay. All inference uses zero GPU layers.

## AirLLM + Hermes

AirLLM **4.0.0** has its own dependency directory and **CPU-only PyTorch 2.6.0**. The local
adapter exposes its generation through a loopback API and converts native `<tool_call>`
responses into real Hermes tool calls. Hermes executes the tools, returns their results
to the model, and continues the conversation. No webpage or cloud service is involved.
The default llama.cpp path and its GGUF model selection remain available.

Open **menu 9**:

1. Start interactive Hermes with the selected AirLLM model.
2. Run/resume `tasks.txt` with AirLLM, then close processes and exit the launcher.
3. Download/select a model using a public Hugging Face **repository** URL, for example
   `https://huggingface.co/Qwen/Qwen3-1.7B`.
4. Select a downloaded model or enter an existing local Transformers directory.
5. Edit `tasks.txt`.
6. Show progress for the AirLLM queue.
7. Repair AirLLM dependencies from the cached wheels.

The installed Qwen3-1.7B is a small starting model for this path. In validation it could
execute terminal/file actions, but sometimes mishandled exact file contents and review
instructions. Use a more capable instruction model for demanding autonomous work. Downloads and preparation
can take several minutes, especially the first split into layer files. Later launches
reuse those files. Models must have an instruction/chat template; a model's ability to
generate text does not establish reliable tool use. Unknown or malformed tool calls,
invalid argument types, and calls in truncated output are rejected instead of executed.
Choose at least 8,192 context tokens to leave room for tool definitions and the task.
The adapter parses Qwen/Hermes-style `<tool_call>` output; other model families are not
validated here and must follow that requested format to execute tools.

Direct launch, with the selected AirLLM model:

```bash
bash launch.sh --backend airllm --chat
bash launch.sh --backend airllm --run --tasks tasks.txt
```

Use `--airllm-model /path/to/checkpoint` to override selection for one run. Queue runs
use `state/airllm-task-loop.json`, separate from the default GGUF queue; an explicit
`--state` can choose another progress file. Both paths use the same `tasks.txt`,
`prompt.txt`, USB workspace and independent reviewer. Completed AirLLM queue runs exit
the menu and shut down their owned model/agent processes. Ctrl+C interrupts and preserves
progress. The service log is `logs/airllm-server.log`.

AirLLM reads Transformers weights, **not GGUF**. It keeps originals and creates layer
files under `airllm/layers/`. Downloads pin the repository revision, verify weight hashes
and check space for originals, layers and a reserve. Inference uses CPU float32, without
CUDA packages or compression that requires CUDA. After downloading, inference is offline.
The CPU adapter leaves per-layer tensor unloading intact but runs expensive Python
garbage collection once per forward pass and at request cleanup, rather than after every
layer. Generation progress and timings are recorded in the service log.
Disconnected inference clients cancel generation at the next generated token, freeing
the server for a fresh task session. An in-progress prompt pass may need to finish first.
An optional bounded CPU weight cache avoids repeatedly reading and converting admitted
layers. `cache_gb` defaults to an 8 GiB ceiling and is reduced automatically to leave
headroom for active layers, KV context memory and the OS. Layers outside the budget keep
streaming; quantized checkpoints bypass this cache. Set `cache_gb` to 0 for minimum
weight residency. Cache tensors are in RAM and are released when the service exits.

`airllm-config.json` controls the selected Transformers model, context and generation
length, toolsets, plus longer request/startup/session timeouts. For small CPU models, the
default `"toolsets": ["terminal"]` exposes terminal and process tools; Hermes reads,
creates and edits files by actually executing shell/Python commands. This reduces prompt
size and tool-selection confusion. With a capable model, set it to
`["terminal", "file", "todo"]` for dedicated file/patch/search and planning tools.
Local sessions use compact role prompts for workers and independent reviewers, retaining
the USB's task instructions and Hermes's actual tool execution. This avoids mixing the
SDK's general execution brief into the review role and reduces CPU prompt processing.
The default context is 16,384 tokens,
capped to the model's advertised maximum. The adapter checks actual token counts and
rejects overflow rather than silently truncating task instructions. In AirLLM sessions
and local GGUF sessions, a process-local SDK compatibility setting lowers Hermes's usual context floor;
no vendored Hermes code is edited and no fictional 64K context is advertised. Smaller
contexts and models can reduce reliability on long tasks; the persistent outer loop
starts fresh sessions when needed.

AirLLM reduces resident weight memory by repeatedly loading layers from disk. It does
not remove weight storage, activation/context memory, or computation. Uncompressed 70B
weights alone are about 140 GB, exceeding this 59 GB drive before layer copies. Huge
models on a CPU and USB are consequently not a practical default here. Performance and
tool reliability need measuring for each selected model; integration does not guarantee
that arbitrary huge models will fit or finish tasks efficiently. The old standalone
`airllm_experiment.py` prompt/synthetic test remains available as a diagnostic CLI.

## Portable Ubuntu toolkit

`toolkit/` contains a compact, terminal-only assessment set built from Ubuntu packages:

- discovery and clients: Nmap/Ncat, Netcat, iproute2, ping, Curl, DNS, SMB/RPC,
  LDAP, NetBIOS and SNMP tools;
- password auditing: Hashcat 7.1.2 with bundled PoCL CPU OpenCL, John the Ripper,
  Hydra, Crunch and `pw-inspector`;
- content discovery: Gobuster and FFUF;
- local inspection: `getcap`, `getpcaps`, `capsh`, `lsof`, `file`, `strings`,
  `readelf`, `objdump`, `jq` and BusyBox.

The official Kali `wordlists` 2026.2.0 package supplies an uncompressed
`toolkit/share/wordlists/rockyou.txt` (139,921,507 bytes). Its original package is kept
in the checksum-pinned repair cache. Use these tools and password candidates only on
systems and accounts you are authorized to assess.

Hermes puts `toolkit/bin` on its PATH automatically, for both backends and task/chat
sessions. No host `apt install` is needed during a normal launch.

Each wrapper uses the USB's private dynamic loader and shared libraries, rather than
injecting toolkit libraries into Python or the model server. Nmap's probes, service
signatures, scripts and other data are included, along with a CA bundle and license files.
All symlinks are materialized as regular files for exFAT. The manifest records Ubuntu
package versions and SHA-256 values checked against signed Ubuntu indexes. A separate
manifest pins the official Kali data package by SHA-256. Cached `.deb` files allow
verified offline repair:

```bash
hermes/python/bin/python3.12 -s -B setup_toolkit.py --offline
```

Repair uses temporary space in the host's `/tmp` and its essential `dpkg-deb` command;
it installs no packages and runs no package-maintenance scripts. Normal toolkit execution
reads its executables, libraries and data from the USB. To use the same tools yourself:

```bash
source toolkit/env.sh
nmap --version
```

**Menu 10** shows tools, runs a self-test against a temporary **127.0.0.1-only** HTTP
service, or repairs the bundle. The self-test checks real Nmap open-port/service detection,
Netcat/Ncat connections, Curl responses, passive interface information, CPU OpenCL and a
synthetic two-candidate Hashcat recovery. It does not scan the connected LAN. Its receipt
is `logs/toolkit-self-test.json`. Hashcat's first CPU kernel build may take several
seconds; compiled kernels are cached under `state/tool-home/` on the USB.

John's Debian build expects a per-user `.john/john.conf`. On its first invocation the
wrapper creates that file in the logged-in user's home only when it is absent; existing
John configuration is preserved. John result/session files consequently use that user's
normal `.john` folder.

For the request “list machines and their services on this computer's network,” the agent
can inspect `ip -j address show` and `ip -j route show`, determine the requested connected
subnet, then use Nmap and save XML/text evidence under `workspace/`. TCP connect scanning
(`-sT`) and service detection (`-sV`) work as an ordinary user. Raw SYN/ARP/UDP operations
and packet capture may need root/capabilities supplied by Ubuntu; exFAT cannot carry Unix
file capabilities. YOLO cannot provide those privileges. Filtered/offline hosts and
non-default ports mean an inventory cannot promise every machine or every running service.

To grow this kit, add selected Ubuntu packages and their dependency closure to the signed,
hash-pinned cache, then rebuild the regular-file bundle and test its commands on clean
Ubuntu. The current set occupies about 1.9 GB including caches and RockYou. Tools with
hard-coded system paths, Python runtimes, kernel modules, or required capabilities need
individual packaging work. The installed binaries and private loader were tested in an
Ubuntu 26.04.1 x86_64 userland; other Ubuntu releases or CPUs have not been physically
tested. Tools still run with the logged-in user's permissions: this bundle does not grant
sudo, access to protected password hashes, packet-capture rights or raw-socket capabilities.

## Backup and validation

The old program, CAI environment, task files, state, logs and workspace were backed up to
`/home/charles/Work/CyberCapstone-backup-2026-10-05` before replacement. Model weights were
not duplicated into that backup; they remain on the USB. See `BACKUP-NOTES.json` there.
CAI launchers and packages were removed after installing the new bundle. Existing models,
task content, work files and old progress/log files were preserved.

See `VALIDATION.md` for the actual checks and their scope.

Upstream references: [Hermes release](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.9.24),
[Qwen model](https://huggingface.co/Qwen/Qwen3.5-9B),
[GGUF quantization](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF),
[AirLLM](https://github.com/lyogavin/airllm),
[Ralph loop inspiration](https://github.com/KLIEBHAN/ralph-loop).
