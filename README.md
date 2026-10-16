# USBAI-Cyber-Capstone

- `Prototype1_demo1/`: original prototype, preserved unchanged from Git HEAD.
- `prototype2_llm_fit/`: updated offline llmfit integration and model drop folder.

Strict offline Linux startup: `bash linux.sh`.
See [USB preparation and offline model selection](prototype2_llm_fit/OFFLINE_MODELS.md)
for required binaries, model manifest fields, memory policy and validation.

Model weights and generated runtimes are ignored by Git. Push/pull transfers the
code and manifest; copy model files separately into
`prototype2_llm_fit/models/drop-in/` on the USB/SD card.
