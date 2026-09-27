"""The evaluation must never write into a trained student's folder, and must read it exactly as training did."""
import json
import os

import pytest

from nanodistill.bfcl_local import REFERENCE_FILES, compatible_copy

# a student config as transformers 5 saves it: the RoPE base lives in rope_parameters
STUDENT_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
                  "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                  "num_hidden_layers": 2, "vocab_size": 1000, "max_position_embeddings": 32768,
                  "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}, "dtype": "bfloat16"}


def make(tmp_path):
    student, reference = tmp_path / "student", tmp_path / "reference"
    student.mkdir(); reference.mkdir()
    (student / "model.safetensors").write_bytes(b"trained weights")
    (student / "config.json").write_text(json.dumps(STUDENT_CONFIG))
    (student / "generation_config.json").write_text(json.dumps({"eos_token_id": 151643, "max_new_tokens": 2048}))
    # a snapshot folder may hold more than we ask for, e.g. weights cached by an earlier baseline run
    (reference / "model.safetensors").write_bytes(b"someone else's weights")
    for f in REFERENCE_FILES:
        (reference / f).write_text(json.dumps({"from": "reference"}))
    return student, reference


def test_compatible_copy_leaves_the_student_untouched(tmp_path):
    student, reference = make(tmp_path)
    out = compatible_copy(str(student), str(reference), out=str(tmp_path / "copy"), max_len=12288)
    assert (student / "model.safetensors").read_bytes() == b"trained weights"
    assert open(os.path.join(out, "model.safetensors"), "rb").read() == b"trained weights"
    assert not os.path.islink(os.path.join(out, "model.safetensors"))
    assert json.load(open(student / "config.json")) == STUDENT_CONFIG
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["torch_dtype"] == "bfloat16" and config["max_position_embeddings"] == 12288
    assert config["rope_theta"] == 1000000 and config["rope_scaling"] is None
    # every 0.6B model is evaluated with Qwen3-0.6B's tokenizer and generation settings
    assert json.load(open(os.path.join(out, "generation_config.json"))) == {"from": "reference"}


def test_the_evaluation_library_reads_qwen3s_rope_base(tmp_path):
    transformers = pytest.importorskip("transformers")
    student, reference = make(tmp_path)
    out = compatible_copy(str(student), str(reference), out=str(tmp_path / "copy"))
    config = transformers.AutoConfig.from_pretrained(out)
    if int(transformers.__version__.split(".")[0]) < 5:     # the eval environment (vLLM 0.8.5)
        assert config.rope_theta == 1000000
        assert transformers.AutoConfig.from_pretrained(str(student)).rope_theta == 10000   # the bug, unfixed
    else:
        assert config.rope_parameters["rope_theta"] == 1000000


def test_direct_mode_is_the_handler_prompt_plus_the_empty_think_block():
    pytest.importorskip("bfcl_eval")
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    from nanodistill.bfcl_runner import DirectQwenFCHandler
    from nanodistill.data import EMPTY_THINK

    messages, tools = [{"role": "user", "content": "Weather in Hanoi?"}], [{"name": "f", "description": "d"}]
    direct = DirectQwenFCHandler.__new__(DirectQwenFCHandler)
    assert direct._format_prompt(messages, tools) == QwenFCHandler._format_prompt(None, messages, tools) + EMPTY_THINK
