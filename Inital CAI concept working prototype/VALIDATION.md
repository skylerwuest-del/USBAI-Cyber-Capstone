# Validation on the CapStone USB

Checks performed on 2026-10-01 at `/run/media/charles/CapStone`.

- All 25 project tests passed; output is in `logs/unit-tests.txt`.
  Four additional typed CAI adapter tests passed using the bundled Python and
  dependencies; output is in `logs/cai-adapter-tests.txt`.
- Downloaded the official llama.cpp `b11146` Ubuntu x86_64 CPU archive and
  Qwen2.5-Coder-7B-Instruct Q4_K_M. File sizes and published SHA-256 values matched
  `assets.json` before installing them.
- The terminal menu opened successfully. The Hugging Face API returned the GGUF
  metadata/checksum used by the downloader. Tests exercise repository/direct URLs,
  download-and-select behavior, and split-file grouping with mocked download bytes.
- Atomic checkpoints, directory fsync, and advisory locking worked on the actual
  exFAT filesystem.
- Real model inference through `bash launch.sh --run` created `workspace/hello.py`,
  executed it through bash, and passed the task's independent Python verification.
  The sample completed in three tool steps: write, shell, finish.
  See `logs/smoke-run.txt`, `logs/actions.jsonl`, and `state/progress.json`.
- Launching the completed queue again skipped the saved completed task without
  starting another model process; see `logs/resume-test.txt`.
- A live model server was killed with SIGKILL. The supervisor restarted it and its
  health endpoint returned success. Clean shutdown released the listening port;
  see `logs/server-recovery-test.txt`.
- All project Python source parses using Python 3.10 grammar. Runtime tests used
  the current host's Python 3.14.7.

The current host is Omarchy/Arch Linux, x86_64, with 32 GB RAM. The real model tests
used CPU inference with zero GPU layers. The Ubuntu binary ran successfully here.
An Ubuntu host/container was not available for execution testing: local Docker
access required a sudo password. The official upstream release workflow builds
this x86_64 runtime on Ubuntu 22.04. Ubuntu dependency/setup instructions are in
`README.md`; target-machine behavior still needs confirmation on that computer.

The example task is already marked completed and its output remains in the USB
workspace. Edit the queue or give a task a new ID to run new work. No agent or model
server was left running after validation.

## CAI integration

- Installed hash-verified public `cai-framework==0.5.10` and portable CPython
  3.12.14 on the USB, with 123 dependency wheels pinned by hashes in
  `cai-requirements.lock`. Binary wheels were selected for glibc <= 2.35 and
  CPython 3.12, targeting x86_64 Ubuntu 22.04 and newer. The unused LiteLLM proxy
  extra is omitted because its Rich constraint conflicts with CAI's own metadata.
- The actual CAI CLI imports with the generated local endpoint settings;
  `logs/cai-import-test.txt` records this check.
- Real CAI SDK inference with `openai/capstone` invoked `generic_linux_command`
  and created `workspace/cai/cai-smoke.txt`. An independent exact-content check
  passed. See `logs/cai-smoke-test.txt`.
- Native tool calls are preserved. The local non-streaming adapter accepts only
  whole JSON objects naming a declared function whose arguments validate against
  that function's schema. Invalid argument types, missing required fields, extra
  properties, other model names, and truncated output are not converted.
- `bash startup.sh` was opened in a real PTY. It started the USB model, showed the
  CAI prompt with `CAI_MODEL=openai/capstone`, and exited with code 0 after `/exit`.
  The owned model process also exited. No CAI/model processes were left running.
- CAI history was saved at `state/cai/history.txt` on exFAT. Telemetry and tracing
  are disabled in generated settings. Local pricing and tokenizer caches avoid
  mandatory remote pricing/tokenizer retrieval on subsequent launches.
- Host cloud settings and implicit `.env` loading cannot override the generated
  local endpoint at startup. Tests also cover first-time install ordering, model
  readiness before CAI opens, model recovery while CAI is running, and terminating
  the owned CAI child on interruption.

CAI mode is interactive and does not run CapStone's checkpointed JSON/text task
queue. The original queue remains available through `bash launch.sh --run`.
Ubuntu itself has not been available for execution testing; the target-host
limitation described above still applies to this new bundle.

## Terminal-only operation

- Compared Portable Local Studio's exFAT handling with the existing bundle.
  CapStone already materializes interpreter/library links as ordinary files and
  installs Python dependencies without a virtual environment. No Node.js
  dependency workaround was needed.
- The launcher now passes `--no-webui` to the bundled llama.cpp runtime. A live
  check returned HTTP 404 for `/`, HTTP 200 for `/v1/models`, and a successful
  real CPU response from `/v1/chat/completions`. The owned server shut down
  cleanly. See `logs/terminal-only-test.txt`.
- All 25 project tests still passed after this change (`logs/unit-tests.txt`).

## Preordained task menu and automatic exit

- Menu option 7 opens `tasks.txt` in a terminal editor, preserves its previous
  contents in `tasks.txt.bak`, validates the saved queue, and holds the session
  lock while editing. Option 8 runs that queue through the existing persistent
  worker and exits the menu/launcher after all tasks verify and the worker has
  shut down its model. A stopped or unsuccessful run returns to the menu.
- All 27 project tests passed, including two download durability checks. A failed
  file flush leaves the previous destination and complete partial intact.
- `tests/menu_runtime_check.py` exercised editing with a filename containing
  spaces, backup preservation, rejecting edits during an active session, and
  rejecting an invalid queue before model startup. It then ran two real tasks
  with the USB model on CPU in an isolated workspace on the same exFAT drive.
  Both outputs passed independent exact-content checks; the model PID exited,
  its port was released, and the menu exited without another prompt. Repeating
  option 8 skipped the completed tasks and exited without starting the model.
  See `logs/menu-runtime-check.txt`, `logs/menu-smoke-run.txt`, and
  `logs/menu-smoke-actions.jsonl`. User task progress was preserved.
- The initial live test exposed unusable model responses. The installed model's
  checksum no longer matched `assets.json` or pinned Hugging Face metadata.
  The kernel also reported an improperly unmounted exFAT volume; the cause of
  the model corruption was not established. Downloaded the pinned model again
  and verified its checksum twice, including after fsync and evicting its file
  pages from cache. The subsequent real task test succeeded. Details are in
  `logs/model-repair.json`. The downloader now fsyncs downloaded bytes before
  checksum acceptance and fsyncs the parent after publishing the file.
- CAI's actual CLI imports successfully after the model repair.

The target Ubuntu computer still needs execution testing. Safely eject the USB
after CapStone exits before physically disconnecting it. No AI processes were
left running after these checks.
