"""xLAM function-calling examples in exactly the format BFCL shows a Qwen3 model at test time.

Training and evaluation must see the same text: the same tool schema (BFCL's "dict"-style JSON with its
Python-syntax hint), the same system prompt, and the same <tool_call> output format. `prompt` mirrors
bfcl_eval's QwenFCHandler._format_prompt for a single user turn; tests/test_bfcl_format.py checks it
against the handler itself.
"""

import json
import random
import re

DATASET = "Salesforce/xlam-function-calling-60k"
PYTHON_HINT = " Note that the provided function is in Python 3 syntax."   # BFCL appends this to every description
EMPTY_THINK = "<think>\n\n</think>\n\n"                                    # Qwen3's non-thinking assistant prefix

_SCALAR = {"str": "string", "int": "integer", "float": "float", "bool": "boolean"}
_TOOL_CALL = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)


def _bfcl_type(xlam_type: str) -> dict:
    """xLAM's Python-style type string ('List[int]', 'str, optional') as a BFCL JSON-schema fragment."""
    base = xlam_type.split(",")[0].strip()
    if base in _SCALAR:
        return {"type": _SCALAR[base]}
    head = base.split("[")[0].lower()
    if head in ("list", "set"):
        inner = base[base.find("[") + 1:-1].strip() if "[" in base else ""
        return {"type": "array", "items": {"type": _SCALAR[inner]}} if inner in _SCALAR else {"type": "array"}
    if head == "tuple":
        return {"type": "tuple"}
    if head == "dict":
        return {"type": "dict"}
    return {"type": "any"}


def bfcl_tool(tool: dict) -> dict:
    """One xLAM tool as a BFCL function document (after BFCL's own Python-hint preprocessing)."""
    properties, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        properties[name] = {**_bfcl_type(str(spec.get("type", "str"))), "description": spec.get("description", "")}
        if "optional" not in str(spec.get("type", "")) and "default" not in spec:
            required.append(name)
    return {"name": tool["name"], "description": tool.get("description", "") + PYTHON_HINT,
            "parameters": {"type": "dict", "properties": properties, "required": required}}


def prompt(tools: list[dict], query: str) -> str:
    """The prompt BFCL's QwenFCHandler builds for one user turn, ending at the assistant header."""
    text = ""
    if tools:
        text += "<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        text += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools:
            text += f"\n{json.dumps(tool)}"
        text += ('\n</tools>\n\nFor each function call, return a json object with function name and arguments '
                 'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, '
                 '"arguments": <args-json-object>}\n</tool_call><|im_end|>\n')
    return text + f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


def completion(calls: list[dict] | None = None, reasoning: str = "", text: str = "") -> str:
    """What the student learns to write after the assistant header: a think block, then calls or text."""
    body = "\n".join("<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}) + "\n</tool_call>"
                     for c in (calls or []))
    think = f"<think>\n{reasoning.strip()}\n</think>\n\n" if reasoning.strip() else EMPTY_THINK
    return think + (body if calls else text) + "<|im_end|>"


def parse_calls(output: str) -> list[dict]:
    """Tool calls in a model output, the way BFCL's QwenFCHandler extracts them (bad JSON is dropped)."""
    calls = []
    for match in _TOOL_CALL.findall(output.split("</think>")[-1]):
        try:
            call = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and "name" in call and isinstance(call.get("arguments"), dict):
            calls.append({"name": call["name"], "arguments": call["arguments"]})
    return calls


def _canonical(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def same_calls(a: list[dict], b: list[dict]) -> bool:
    """Equal as multisets of (name, arguments); 5 and 5.0 count as equal."""
    key = lambda calls: sorted(json.dumps(_canonical(c), sort_keys=True) for c in calls)
    return key(a) == key(b)


def load_examples(dataset=None) -> list[dict]:
    """Every xLAM example as {id, query, tools (BFCL format), calls}."""
    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset(DATASET, split="train")
    return [{"id": int(r["id"]), "query": r["query"],
             "tools": [bfcl_tool(t) for t in json.loads(r["tools"])],
             "calls": json.loads(r["answers"])} for r in dataset]


def split(examples: list[dict], n_train: int, n_irrelevant: int, banned_names: set[str], seed: int = 0) -> dict:
    """A fixed training sample, plus irrelevance cases built from other examples.

    Examples whose tools share a name with any BFCL function are left out, so BFCL's tools stay unseen.
    An irrelevance case removes every tool the answer calls; the remaining tools do not fit the query,
    and the right behaviour is to answer without calling any.
    """
    clean = [e for e in examples if not {t["name"] for t in e["tools"]} & banned_names]
    rng = random.Random(seed)
    rng.shuffle(clean)
    train, rest = clean[:n_train], clean[n_train:]
    irrelevant = []
    for e in rest:
        used = {c["name"] for c in e["calls"]}
        left = [t for t in e["tools"] if t["name"] not in used]
        if left:
            irrelevant.append({"id": e["id"], "query": e["query"], "tools": left, "calls": []})
        if len(irrelevant) == n_irrelevant:
            break
    return {"train": train, "irrelevant": irrelevant, "excluded_for_overlap": len(examples) - len(clean)}
