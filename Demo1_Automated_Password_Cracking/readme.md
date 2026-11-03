# Demo 1 — Automated Local Password Audit

> **Outdated snapshot:** This demo preserves an older version of the project.
> The current application is maintained in [Source/](../Source/). This demo does
> not include all current features, fixes, or setup changes.

This demonstration uses the CapStone USB AI assistant to carry out a prewritten
password-audit task on an authorized Ubuntu lab machine. A local language model
coordinates the work through Hermes terminal and file tools; John the Ripper
performs the password guessing against the bundled RockYou wordlist.

The aim is to demonstrate a complete agent workflow: inspect the machine, execute
tools, preserve long-running work, check the results, and write an evidence-based
report. The interface is entirely in the terminal. Installed inference runs locally.

## Folder layout

The application bundle is inside:

```text
Demo1_Automated_Password_Cracking/
├── readme.md
└── Make this into an IMG or paste to USB/
    ├── launch.sh                  # Normal entry point
    ├── config.json                # Model, workspace and timeout settings
    ├── tasks.txt                  # Prewritten password-audit request
    ├── prompt.txt                 # Standing instructions for the agent
    ├── task_loop.py               # Worker/reviewer queue and completion checks
    ├── long_job.py                # Persistent command supervisor
    ├── models/                    # GGUF weights must be supplied here
    ├── hermes/                    # Bundled agent and Python dependencies
    ├── runtime/                   # Local llama.cpp model server
    ├── toolkit/                   # Assessment tools, libraries and wordlists
    ├── state/                     # Queue history, durable jobs and runtime state
    ├── logs/                      # Agent observations and model output
    └── workspace/                 # Task files and audit findings
```

This is an outdated, packaged demo snapshot preserved for reference and
demonstration. Use the repository's [Source/](../Source/) directory for current
development and the latest application. Changes in `Source/` do not automatically
update this snapshot.

## Requirements and current snapshot status

- An authorized Ubuntu x86_64 lab computer. The original target has 32 GB RAM
  and runs inference on its CPU without a GPU.
- Sufficient storage for the installed model and application bundle.
- Execution permission for the bundled Python, server, loaders and tool wrappers.
  On exFAT, the drive's mount options must allow execution.
- Access to the password hashes being audited. An ordinary Ubuntu user normally
  cannot read `/etc/shadow`; an administrator must provide appropriate access.
  Hermes YOLO mode approves supported tool actions but does not grant root.

**As inspected on November 3, 2026:** bundled Python, the model-server executable,
John's wrapper and RockYou are present. However, the configured
`models/Qwen3.5-9B-Q4_K_M.gguf` file is missing and no model files were found in
this snapshot's `models/` directory. Supply a compatible model before running the
audit. The existing `state/task-loop.json` also records a completed task; this is
not a blank first-run bundle. The expected `workspace/password_audit_findings/`
directory was absent at inspection. Saved completion text alone is not a substitute
for the original evidence files or a new validation run.

## Prepare and launch

For a USB deployment, copy the **contents** of
`Make this into an IMG or paste to USB/` onto the intended USB filesystem, keeping
the relative directory structure intact. `launch.sh` should be at the root of
that application copy.

The folder name does not make this a raw disk image or a bootable operating
system. Creating an `.img` requires a separate filesystem-image preparation step.
The Ubuntu computer runs the program from the mounted storage.

From this Git repository's root, open the demo bundle:

```bash
cd "Demo1_Automated_Password_Cracking/Make this into an IMG or paste to USB"
bash launch.sh
```

Use menu **5** to download a GGUF model from a Hugging Face URL, or put compatible
weights in `models/` and use menu **6** to select them. Downloading requires
internet access; installed local inference does not. This demo defaults to a
9B model. Small models are useful for launch tests but can be unreliable at
multi-step tool use and independent review.

Then:

1. Choose **2** to inspect or edit the preordained task in `tasks.txt`.
2. Choose **3** to run or resume that task and exit after accepted completion.
3. Inspect the saved evidence and report, not only the agent's final message.

The equivalent queue command is:

```bash
bash launch.sh --run --tasks tasks.txt
```

`launch.sh` is the everyday entry point. `setup.sh` and the setup Python scripts
are installation/repair helpers; some depend on bundled caches. This snapshot
should not be assumed to bootstrap every missing dependency from a bare clone.

## What the task requests

The included `tasks.txt` asks the assistant to:

- Record system metadata, privileges and the local account inventory.
- Obtain accessible password hashes and distinguish usable hashes from locked
  accounts or accounts without password hashes.
- Prepare username-preserving input for bundled John using `--format=crypt`.
- Run the complete RockYou wordlist through a named, resumable durable job.
- Save commands, logs, cracked/uncracked account lists and remediation findings
  under `workspace/password_audit_findings/`.
- Keep plaintext recovery results in a separate results file rather than the
  summary report, and report concrete permission blockers honestly.

RockYou is located at `toolkit/share/wordlists/rockyou.txt`. A completed wordlist
audit does not guarantee recovery of every password: passwords absent from that
list can remain uncracked. CPU auditing of expensive hashes can take days.

## Task continuity, completion and reruns

The worker acts through real tools. A separate reviewer inspects the results.
Additional completion checks require a successful durable John job, a loaded-hash
log entry and John's `Session completed` marker. Account coverage and the actual
results still need inspection; a completion marker is not proof of every claim
in a generated report.

Long commands are managed through `capstone-job`, with records in `state/jobs/`.
The AI should inspect the same job across session transitions rather than start
the wordlist over. Tool-specific recovery files support deliberate resumption.

Menu 3 normally **resumes** progress. An unchanged accepted task can be skipped;
missing durable evidence can cause it to be reopened. Before a genuinely fresh
demonstration, stop any active jobs and preserve the prior generated state and
results in a separate backup, preferably working from a separate demo copy.
Resetting only the queue progress file does not clear durable jobs or John's
previous results. Do not delete state belonging to a job that is still running.

Ctrl+C interrupts the agent/model runner. Durable jobs may remain active and
must be inspected or deliberately stopped. An impossible task or unavailable
permission can keep the retry loop open instead of producing a successful exit.

## Evidence and sharing

Inspect `state/task-loop.json`, `state/jobs/`, `logs/sessions/` and the generated
workspace report to establish what actually ran. This README was prepared by
inspecting the snapshot; no new password audit was executed to write it.

The local demo folder can contain an installed runtime snapshot. Git publishes
its source, settings and documentation; the demo-specific ignore rules keep installed
dependencies, model weights, caches, state, logs and private audit artifacts local.
A GitHub clone is therefore not a complete, ready-to-flash USB image. Supply the
required runtime assets separately before preparing an offline demonstration.
Review any additional files before publishing them; audit data may retain account
details or recovered credentials.

See the bundle's [application README](Make%20this%20into%20an%20IMG%20or%20paste%20to%20USB/README.md)
and [validation history](Make%20this%20into%20an%20IMG%20or%20paste%20to%20USB/VALIDATION.md)
for the underlying CapStone implementation and historical checks.
