#!/usr/bin/env python3
"""Entry point using the USB's portable Python and hash-pinned CAI packages."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CAI = ROOT / "cai"


def configure():
    sys.path.insert(0, str(CAI / "site-packages"))
    settings = json.loads((CAI / "local-config.json").read_text())
    os.environ.update(settings)
    import litellm
    from cai.sdk.agents import set_tracing_disabled
    set_tracing_disabled(True)
    from cai_tool_adapter import install
    install()
    # Register local context/cost metadata so accounting does not query cloud prices.
    info = {"max_tokens": int(settings["CAPSTONE_CONTEXT_SIZE"]),
            "max_input_tokens": int(settings["CAPSTONE_CONTEXT_SIZE"]),
            "max_output_tokens": int(settings["CAPSTONE_MAX_TOKENS"]),
            "input_cost_per_token": 0, "output_cost_per_token": 0,
            "litellm_provider": "openai", "mode": "chat", "supports_function_calling": True}
    litellm.register_model({"capstone": info, "openai/capstone": info})
    # CAI 0.5.10 hard-codes ~/.cai for history. Override its history provider at
    # runtime, preserving upstream package files and keeping history on the USB.
    import cai.repl.ui.logging as history
    path = ROOT / "state" / "cai" / "history.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    history.setup_session_logging = lambda: path
    # Public CAI expects a source checkout for its banner version and queries
    # a remote model catalogue. Use installed metadata and the one local model.
    import importlib.metadata
    import cai.repl.ui.banner as banner
    banner.get_version = lambda: importlib.metadata.version("cai-framework")
    banner.get_supported_models_count = lambda: 1
    return settings


async def smoke_test(settings):
    from openai import AsyncOpenAI
    from cai.sdk.agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
    from cai.tools.reconnaissance.generic_linux_command import generic_linux_command
    result_file = Path.cwd() / "cai-smoke.txt"
    result_file.unlink(missing_ok=True)
    client = AsyncOpenAI(api_key=settings["OPENAI_API_KEY"], base_url=settings["OPENAI_BASE_URL"],
                         timeout=float(settings["CAPSTONE_REQUEST_TIMEOUT"]))
    agent = Agent(name="CapStone local integration check", instructions=
            "Use generic_linux_command once to run: printf 'CAI local model OK\\n' > cai-smoke.txt. "
            "Then reply Done. Do not run any other command.",
            tools=[generic_linux_command],
            model=OpenAIChatCompletionsModel(model=settings["CAI_MODEL"], openai_client=client),
            model_settings=ModelSettings(temperature=0.1, max_tokens=300, parallel_tool_calls=False))
    try:
        await Runner.run(agent, "Create cai-smoke.txt using the tool now.", max_turns=4)
        if result_file.read_text() != "CAI local model OK\n":
            raise RuntimeError("CAI tool result failed its independent verification")
        print("CAI actual local inference and shell tool verified.")
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    settings = configure()
    if args.smoke_test:
        asyncio.run(smoke_test(settings))
        return 0
    from cai.cli import main as cai_main
    if args.check:
        print("CAI CLI imported successfully with USB-local model settings.")
        return 0
    sys.argv = ["cai"] + ([args.prompt] if args.prompt is not None else [])
    cai_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
