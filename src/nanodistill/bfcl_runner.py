"""Run the bfcl CLI with one extra model registered: `python bfcl_runner.py NAME HF_NAME MODE bfcl-args...`.

MODE "think" uses BFCL's Qwen3 FC handler as is; "direct" appends the empty think block to its prompt
(Qwen3's non-thinking mode). Runs inside the Modal evaluation image, and locally in the eval environment.
"""
import sys

from overrides import override

from bfcl_eval.constants import model_config as mc
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class DirectQwenFCHandler(QwenFCHandler):
    """Qwen3 in non-thinking mode: the prompt ends with the empty think block."""

    @override
    def _format_prompt(self, messages, function):
        return super()._format_prompt(messages, function) + "<think>\n\n</think>\n\n"


def register(name: str, hf_name: str, mode: str) -> None:
    base = mc.MODEL_CONFIG_MAPPING["Qwen/Qwen3-0.6B-FC"]
    handler = DirectQwenFCHandler if mode == "direct" else QwenFCHandler
    mc.MODEL_CONFIG_MAPPING[name] = type(base)(**{**base.__dict__, "model_name": hf_name, "display_name": name,
                                                  "model_handler": handler})


if __name__ == "__main__":
    name, hf_name, mode = sys.argv[1:4]
    register(name, hf_name, mode)
    from bfcl_eval.__main__ import cli
    sys.argv = ["bfcl"] + sys.argv[4:]
    cli()
