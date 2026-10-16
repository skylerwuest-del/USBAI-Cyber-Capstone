DROP YOUR COMPLETE .gguf MODEL FILES HERE

Then start the application normally with bash linux.sh.
The launcher automatically updates models/models.json before selection.
Finish copying files before starting. No downloads or installs occur.

Dense Llama, Qwen2 and Qwen3 models can be registered automatically.
Unsupported architectures or incomplete files are skipped with an explanation.
Hybrid/recurrent, MoE, split models and adapters need reviewed manual entries.
Models must support at least 4096 context tokens.

Generated memory allowances are conservative estimates, not measured guarantees.
The hardware selector still decides whether the model fits safely.
Deleting an automatically imported file removes its generated manifest entry
on the next startup. Manually maintained manifest entries are preserved.

Optional manual refresh from the project directory:
python3 -B scripts/import_models.py
