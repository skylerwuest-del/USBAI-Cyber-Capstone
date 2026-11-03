# USBAI-Cyber-Capstone

The application source is in `Source/`.

```bash
# First preparation (internet required for missing assets):
bash Source/setup.sh
# Everyday use after setup:
bash Source/launch.sh
```

The startup menu offers offline GGUF fit selection, default/manual launch, and then opens
the original application menu. Choose its option 1 to begin chat. Installed weights live in `Source/models/drop-in/`.
See [application instructions](Source/README.md) and
[validation results and local runtime limitations](Source/VALIDATION.md).
