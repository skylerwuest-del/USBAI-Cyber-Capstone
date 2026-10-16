# USBAI-Cyber-Capstone

The current application lives in `capstone/`. Update this folder in place;
previous versions are available through Git history rather than separate prototype folders.

Strict offline Linux startup: `bash linux.sh`.
See [USB preparation and offline model selection](capstone/OFFLINE_MODELS.md)
for required binaries, model manifest fields, memory policy and validation.

Model weights and generated runtimes are ignored by Git. Push/pull transfers the
code and manifest; copy model files separately into
`capstone/models/drop-in/` on the USB/SD card. Ignored files are not recoverable
from Git history.
