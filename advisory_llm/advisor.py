#!/usr/bin/env python3
"""Defect -> operator recommendation.

Two paths:
  rule_advice()  instant, always available, built from a small table per defect type
  llm_advice()   Llama 3.2 1B Instruct running on the KV260's Arm CPU through the ExecuTorch
                 llama_main runner (see export_llama_executorch.md). Used only when the three
                 environment variables below point at the runner, the .pte model and the tokenizer.

    PCB_LLM_RUNNER     path to llama_main
    PCB_LLM_MODEL      path to the exported .pte
    PCB_LLM_TOKENIZER  path to tokenizer.model

Command line test (no board needed for the rule path):
    python3 advisor.py short 0.82 open_circuit 0.73
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kv260_app"))
from severity import SEVERITY, board_severity  # noqa: E402

ACTION = {
    "short": ("excess copper or etch residue bridging two nets", "rework: cut the bridge and re-test continuity"),
    "open_circuit": ("over-etching, a scratch or a broken trace", "rework with a jumper wire or scrap if it is a power net"),
    "missing_hole": ("drill skipped or broken drill bit", "check the drill program and tool; scrap if no rework drilling"),
    "spurious_copper": ("leftover copper from poor etching or a resist defect", "remove the copper and check the etch line"),
    "spur": ("resist defect or under-etching at the trace edge", "trim if close to a neighbouring net, otherwise pass with note"),
    "mouse_bite": ("irregular etching at the trace edge", "pass with note unless the trace is a high-current net"),
}

SYSTEM = open(os.path.join(HERE, "prompts", "system.txt")).read().strip()
USER = open(os.path.join(HERE, "prompts", "user_template.txt")).read()


def rule_advice(defects):
    if not defects:
        return "No defects found. Pass the board."
    sev = board_severity(defects)
    worst = max(defects, key=lambda d: (["minor", "major", "critical"].index(SEVERITY.get(d["cls"], "minor")), d["conf"]))
    cause, action = ACTION.get(worst["cls"], ("unknown", "inspect manually"))
    decision = {"critical": "Do not ship.", "major": "Hold for rework.", "minor": "Can pass after review."}[sev]
    return f"{worst['cls'].replace('_', ' ')}: likely {cause}. Action: {action}. {decision}"


def llm_available():
    return all(os.path.exists(os.environ.get(k, "")) for k in ("PCB_LLM_RUNNER", "PCB_LLM_MODEL", "PCB_LLM_TOKENIZER"))


def build_prompt(defects):
    lines = "\n".join(f"- {d['cls']} (confidence {d['conf']:.2f}, severity {SEVERITY.get(d['cls'], 'minor')})" for d in defects)
    user = USER.format(verdict="FAIL" if defects else "PASS", severity=board_severity(defects), n=len(defects), defect_lines=lines or "- none")
    # Llama 3 instruct chat format; the runner adds the BOS token
    return (f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")


def llm_advice(defects, seq_len=320, timeout=120):
    if not llm_available():
        return None
    prompt = build_prompt(defects)
    cmd = [os.environ["PCB_LLM_RUNNER"], f"--model_path={os.environ['PCB_LLM_MODEL']}",
           f"--tokenizer_path={os.environ['PCB_LLM_TOKENIZER']}", f"--prompt={prompt}", f"--seq_len={seq_len}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout
    if "assistant<|end_header_id|>" in out:
        out = out.rsplit("assistant<|end_header_id|>", 1)[1]
    out = out.replace("<|eot_id|>", "").strip()
    return " ".join(out.split())[:400] or None


if __name__ == "__main__":
    args = sys.argv[1:]
    defects = [{"cls": args[i], "conf": float(args[i + 1]), "box": ""} for i in range(0, len(args) - 1, 2)]
    print("rule:", rule_advice(defects))
    if llm_available():
        print("llm :", llm_advice(defects))
    else:
        print("llm : not configured (set PCB_LLM_RUNNER, PCB_LLM_MODEL, PCB_LLM_TOKENIZER)")
