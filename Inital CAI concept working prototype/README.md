# CapStone

A USB-local AI worker for x86_64 Ubuntu with 32 GB RAM and no GPU. The USB carries
llama.cpp, GGUF models, task files, agent code, progress, logs, and a workspace.
Inference uses the host CPU and RAM. No cloud inference, cloud API key, Docker,
or system-wide model service is needed. The base worker uses host Python; the
optional CAI harness includes its own Python runtime and dependencies on the USB.

## Launch on Ubuntu

Open a terminal at the root of the USB (its mount path varies by computer):

```bash
cd /media/"$USER"/CapStone
bash launch.sh
```

The menu provides:

1. Start or resume a task queue; starts the model server automatically.
2. Serve the model only, without executing tasks.
3. Download and select a model from a Hugging Face URL.
4. Select a model already on the USB.
5. Show saved task progress.
6. Start the local model and open the CAI harness.
7. Edit preordained tasks in `tasks.txt` using a terminal editor.
8. Run/resume `tasks.txt`, then stop the owned AI processes and exit CapStone.
0. Exit.

For unattended execution, skip the menu:

```bash
bash launch.sh --run
bash launch.sh --run --tasks tasks.txt
```

The runner processes the queue in order. It stops normally **only after all tasks
pass their verification commands**, then shuts down its model server. There is no
step limit or maximum retry count. Ctrl+C or SIGTERM stops the run deliberately;
launch it again to resume. Keep the USB connected and the computer awake while
working. To keep the model server running without the queue:

```bash
bash launch.sh --serve-only
```

The server binds to `127.0.0.1:8080`; the API is at `/v1/chat/completions`.
It restarts if its process dies. Choose another port in `config.json` if occupied.

## Ubuntu prerequisites

Python 3.10+ and bash are needed to run. Internet and curl are needed to download
models/runtime, but the installed model and runner operate offline afterward.
The bundled x86_64 CPU runtime was built by llama.cpp on Ubuntu 22.04 with runtime
CPU variant selection. Ubuntu 22.04 and newer are the intended targets.

If dependencies are missing on Ubuntu, install them once on the target host:

```bash
sudo apt update
sudo apt install python3 curl libstdc++6 libgomp1 libssl3 zlib1g libbrotli1 libzstd1
```

On Ubuntu 24.04+, `libssl3t64` is the corresponding OpenSSL package if `libssl3`
is unavailable. You do not need to run CapStone itself with sudo.

If starting from source files without the model/runtime, or if setup was interrupted:

```bash
bash setup.sh
```

Setup resumes partial downloads and checks pinned SHA-256 hashes in `assets.json`.
Downloads flush their file data and final rename to the USB before reporting
completion. Exit CapStone and safely eject the USB before disconnecting it.
It downloads the CPU runtime and Qwen2.5-Coder-7B-Instruct Q4_K_M (about 4.7 GB).
`bash setup.sh --runtime-only` installs just the runtime. `--model-only` downloads
just the default model. Setup does not change the selected model in `config.json`.

## Choose models from Hugging Face

Select menu option **3** and paste either:

```text
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

A repository URL displays the available GGUF files so you can choose a quantization.
A file URL downloads that model directly. Split GGUF files are downloaded as a
complete set; the first shard is selected. Downloads are pinned to the resolved
repository commit, verified against the published SHA-256, and stored under
`models/<owner>--<repository>--<revision>/` with a `source.json` receipt.

This menu supports **public, ungated GGUF repositories**. It does not convert
Safetensors, PyTorch weights, or other formats. A GGUF file must also be supported
by this version of llama.cpp. For the agent, choose an **instruction/chat model**
that can follow the JSON action protocol. Larger models need more RAM and can be
slower on CPU. Check a downloaded model's license on its repository page.

Changing models is disabled while another runner is using the USB. Use option 4
to switch back to an installed model. Completed tasks stay completed when switching
models; changing a task's contents or ID makes it eligible to run again.

## Launch the CAI harness with the local model

CAI is installed on this USB. For a direct startup entry that automatically starts
both the selected model and the CAI terminal interface:

```bash
bash startup.sh
```

Equivalent commands are `bash launch.sh --cai` and menu option **6**. The launcher
waits for model readiness, opens CAI, supervises model restarts while CAI is open,
and shuts down the owned model server when the CAI process exits. Use CAI's `/exit`
command to close it. Ctrl+C stops the launcher/session. This is a USB launcher;
it does not install an Ubuntu login or boot autostart entry.

An optional initial prompt is passed as one argument:

```bash
bash startup.sh --cai-prompt "Inspect the project in this workspace and describe its files."
```

On each launch the adapter reads `config.json`, so the model selected in the Hugging
Face menu is used automatically. The server exposes that model with the stable
name `capstone`; CAI uses `CAI_MODEL=openai/capstone` and both OpenAI base URL
variables point to `http://127.0.0.1:<configured-port>/v1`. The API key is a local
placeholder. Model inference does not require a cloud account. CAI telemetry,
tracing, and online memory are disabled for this configuration.

CAI starts in `workspace/cai/`, keeps REPL history under `state/cai/history.txt`,
and writes its session artifacts in that USB workspace. Generated local settings
are in `cai/local-config.json` and `cai/.env.local`. The adapter disables implicit
`.env` discovery so another project's cloud settings do not override the local
configuration. CAI still has its own model/agent commands; manually changing those
inside CAI can change its behavior and provider selection.

`config.json` contains `cai.agent`, initially `one_tool_agent` (CAI's terminal-tool
agent). Set this to another agent name supported by CAI if desired. Its maximum
turn count is `inf`. **CAI mode is an interactive harness**, while menu option 1 /
`--run` is CapStone's persistent task queue with independent verification and
checkpoints. CAI does not consume `tasks.json` or inherit that queue's completion
checks. Run the existing queue with `bash launch.sh --run --tasks tasks.json`.

The installation pins the public `cai-framework==0.5.10` package. The upstream
repository is now archived; it also contains a newer professional source tree.
This integration uses the public PyPI release, with portable CPython 3.12.14, so
it does not depend on the target Ubuntu computer's Python package versions.

For first-time installation or repair:

```bash
bash setup-cai.sh
bash setup-cai.sh --offline  # repair from wheels/archive already on the USB
bash launch.sh --cai --check
```

The first `--cai` launch installs the environment automatically if its installation
receipt is missing; that first install needs internet. The already installed
bundle and cached tokenizers work offline. It occupies several GB on exFAT because
its many small files each use a filesystem allocation unit. No symlinks or venv
activation are required. If the USB is mounted `noexec`, its bundled Python binary
needs an executable mount just like `llama-server`.

`cai-assets.json` pins Python and the CAI wheel checksums. All resolved dependency
wheels are cached in `cai/wheels/` and pinned with SHA-256 hashes in
`cai-requirements.lock`. CAI's declared LiteLLM **proxy extra** has a conflicting
Rich version requirement. The installer uses CAI's CLI dependencies without that
unused proxy extra, and pins LiteLLM 1.74.15. A separate LiteLLM proxy server is
not part of this launcher.

LiteLLM is the Python client library CAI uses to send model requests. Here it
connects directly to `http://127.0.0.1:8080/v1` (or your configured local port).
The `openai/capstone` name selects an API format; the configured server runs on
the same computer. Interaction stays in the terminal. The launcher explicitly
passes `--no-webui` to llama.cpp, disabling its bundled browser interface while
keeping the local model API available. No browser is opened.

- [LiteLLM Python SDK and optional proxy](https://docs.litellm.ai/docs/)

The default GGUF has a malformed tool-call example in its embedded chat template
(double braces around a JSON object). The supplied official Qwen template fixes
that example. `config.json` maps this GGUF filename to
`templates/qwen2.5-coder.jinja`; other selected models retain their own templates
unless you add a filename-specific override in `chat_templates`. Some local-model
responses still return a complete JSON tool call as text. The CAI adapter converts
that one exact object into a tool call only if its name was declared in the
request and its arguments pass the tool's JSON schema. Prose, malformed JSON,
unknown tools, and truncated responses stay text. This applies only to the local
`openai/capstone` model in non-streaming mode. An alternative model must follow the
tool-call protocol to work well with CAI.

- [CAI repository and archive status](https://github.com/aliasrobotics/cai)
- [Public CAI 0.5.10 package](https://pypi.org/project/cai-framework/0.5.10/)
- [CAI local endpoint configuration](https://aliasrobotics.github.io/cai/cai/getting-started/configuration/)
- [Portable Python upstream](https://github.com/astral-sh/python-build-standalone)

## Write tasks

For the plain-text queue, run `bash launch.sh` and choose **7** to edit
`tasks.txt`. The menu opens nano, vim, or vi, saves the previous contents as
`tasks.txt.bak`, and checks the saved task format. With nano, save with Ctrl+O,
Enter, and exit with Ctrl+X. A custom terminal editor can be selected with
`CAPSTONE_EDITOR`, for example `CAPSTONE_EDITOR="vim" bash launch.sh`.

Choose **8** to run/resume those tasks. The model starts automatically and keeps
working until each task passes its `VERIFY:` command. After the last verified
task, the runner shuts down its model, cleans up shell tool process groups, and
exits; the menu and launcher exit too. The Ubuntu computer stays on. Already
completed, unchanged tasks are skipped. Errors or a deliberate interruption
return to the menu so the queue can be corrected or resumed. Editing is blocked
while another CapStone run holds the USB session lock.

For a JSON queue, edit `tasks.json`. Each task needs an ID, a prompt describing the work, and a shell
command that proves completion. The command runs in the task's working directory.
Exit code **0** means the check passed. A nonzero exit code or timeout means the
model receives the failure and keeps working.

```json
[
  {
    "id": "create-report",
    "prompt": "Create report.md describing this project and include a Usage section.",
    "verify": "python3 -c \"from pathlib import Path; text=Path('report.md').read_text(); assert 'CapStone' in text and 'Usage' in text\""
  }
]
```

The default working directory is `workspace/` on the USB. To operate on an existing
Ubuntu project, add an absolute `workdir` to that task:

```json
{
  "id": "fix-project",
  "workdir": "/home/alex/projects/example",
  "prompt": "Fix the failing tests. Run the suite and inspect the output.",
  "verify": "python3 -m unittest discover -s tests"
}
```

The USB includes an editable `tasks.txt` and a separate `tasks.example.txt` template.
Write the task in plain
text, add exactly one `VERIFY:` line containing its completion command, and separate
multiple tasks with a line containing `---`. Start it with:

```bash
bash launch.sh --run --tasks tasks.txt
```

The default queue is a small, verifiable demonstration. It creates `workspace/hello.py`.
Do not leave demonstration tasks in a queue intended for different work.

## Terminal access and persistence

The model can execute bash commands and read, write, and list files as the logged-in
user. Commands run without interactive confirmation. This includes host files that
the user can access, when the task calls for it. The USB workspace is the default
working directory, **not an OS sandbox**. Commands get no interactive stdin or
password entry. A task requiring interactive sudo, missing credentials, unavailable
network access, or impossible requirements may keep retrying until the environment
or instructions are corrected.

Completion is determined by your verification command, not the model's claim.
Use a meaningful check for the outcome you actually need. Persistence cannot prove
that an arbitrary task is achievable or that a model will eventually solve it.

Each tool call is checkpointed before execution and after its result. After a crash
in between, the model is told that the pending action may already have run and must
inspect the result before repeating it. This is **at-least-once recovery**, not an
exactly-once guarantee for shell side effects. Durable task IDs, a recent action
history, and progress notes are retained between launches. Recent history is bounded
to keep model context usable during long runs.

Shell commands have a timeout (120 seconds by default) and their child process group
is stopped when they end or time out. Increase `command_timeout` in `config.json` for
long builds or tests. Keep dependent `cd`/environment operations in one shell call;
each call starts a fresh shell. Persistent background daemons are not a task tool.

Paths and options live in `config.json`. Relative paths resolve from the USB root.
`threads: 0` uses half the host's logical CPUs; set a positive value to tune speed.
`context_size`, `max_tokens`, and request/startup timeouts are configurable.

```bash
bash launch.sh --status   # no model starts
bash launch.sh --check    # validate files, saved state, and queue
```

`--check` validates prerequisites, not hardware compatibility or inference quality.

## Files

- `launch.sh`, `menu.py`: interactive menu and unattended launcher.
- `capstone.py`: model process supervision, terminal/file tools, persistent task loop.
- `startup.sh`, `cai_bridge.py`, `cai_bootstrap.py`: managed CAI launch and local model adapter.
- `setup-cai.sh`, `setup_cai.py`, `cai-assets.json`, `cai-requirements.lock`: portable CAI installer and pins.
- `templates/`: model-specific chat templates and source receipt.
- `models_menu.py`: Hugging Face discovery, downloads, and model selection.
- `setup.sh`, `setup.py`, `assets.json`: pinned default assets and checksums.
- `tasks.json`, `tasks.example.txt`: task queue and text format example.
- `models/`, `runtime/`, `licenses/`: model weights, CPU binaries/libraries, licenses.
- `workspace/`: default location for task results.
- `state/progress.json`: checkpoints; do not remove it while a runner is active.
- `logs/actions.jsonl`: action records and bounded output excerpts.
- `logs/server.log`: model startup/inference diagnostics.

Logs accumulate during long runs. Preserve needed evidence and archive old logs
between runs. A single advisory lock prevents two runners from editing the same
progress file concurrently.

## exFAT and noexec mounts

This USB uses exFAT. Use `bash launch.sh` and `bash setup.sh`; they do not depend on
script executable bits. Setup stores shared-library links as regular files because
exFAT cannot create Unix symlinks. The CAI installer extracts bundled Python
on a temporary filesystem that supports links, then copies it to the USB with
links replaced by regular files. Python dependencies install into
`cai/site-packages/` using cached wheels rather than a symlink-based virtual
environment.

[Portable Local Studio's setup](https://github.com/techjarves/Portable-Local-Studio/blob/main/scripts/setup/setup.sh)
uses portable Node.js and substitutes shell wrappers where symlinks are unavailable;
its [Linux launcher](https://github.com/techjarves/Portable-Local-Studio/blob/main/linux.sh)
swaps dependency directories instead of linking them on exFAT. CapStone already
handles the same filesystem limitation for its Python/CAI bundle, so those
Node.js-specific workarounds are not needed here.

If Ubuntu mounts the USB with **noexec**, bash and Python can read the scripts, but
Linux refuses to execute `runtime/llama-server`. Either have the host administrator
mount the USB with execution allowed, or copy only the `runtime/` directory to an
executable host directory and set `server` in `config.json` to its absolute
`llama-server` path. Models, tasks, and state can remain on the USB. The copy option
means runtime execution is from the host, rather than strictly from the USB.

## Validation and upstream references

```bash
python3 -B -m unittest discover -s tests -v
cai/python/bin/python3.12 -B tests/cai_adapter_runtime_check.py
```

Tests exercise verification failures and retries, model/tool errors, checkpoints and
resume, duplicate locks, bounded terminal output, command timeouts, task parsing,
context bounds, Hugging Face URLs, model selection, and split GGUF handling.

- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/b11146/tools/server/README.md)
- [Official Ubuntu runtime build workflow](https://github.com/ggml-org/llama.cpp/blob/b11146/.github/workflows/release.yml)
- [Default Qwen GGUF model repository](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)

See `VALIDATION.md` for checks performed on this USB and their limits.
