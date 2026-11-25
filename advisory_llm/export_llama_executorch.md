# Llama 3.2 1B on the KV260 with ExecuTorch

The advisor runs Llama 3.2 1B Instruct on the KV260's Cortex-A53 cores through ExecuTorch (XNNPACK backend). The DPU stays free for the vision model.

Commands follow the ExecuTorch `examples/models/llama` README. Flags change between ExecuTorch releases, so check that README for the version you install.

## 1. Export on a PC

Download Llama 3.2 1B Instruct (SpinQuant checkpoint) from Meta, then in an ExecuTorch checkout:

```bash
python install_executorch.py
./examples/models/llama/install_requirements.sh

LLAMA_QUANTIZED_CHECKPOINT=path/to/spinquant/consolidated.00.pth
LLAMA_PARAMS=path/to/spinquant/params.json

python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_xnnpack_spinquant.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_QUANTIZED_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}"
```

This gives a 4-bit `.pte` file. Copy it and `tokenizer.model` to the board.

## 2. Build the runner on the KV260 (aarch64 Linux)

```bash
cmake --workflow llm-release
pushd examples/models/llama
cmake --workflow --preset llama-release
popd
```

The binary is `cmake-out/examples/models/llama/llama_main`.

## 3. Test it

```bash
cmake-out/examples/models/llama/llama_main --model_path=llama3_2.pte \
  --tokenizer_path=tokenizer.model --prompt="What causes a short on a PCB?" --seq_len=128
```

## 4. Connect it to the station

```bash
export PCB_LLM_RUNNER=/path/to/llama_main
export PCB_LLM_MODEL=/path/to/llama3_2.pte
export PCB_LLM_TOKENIZER=/path/to/tokenizer.model
python3 advisory_llm/advisor.py short 0.82     # quick check
sudo -E kv260_app/run.sh
```

When these are set, the station shows the rule-based advice at once and replaces it with the Llama answer when it is ready. Without them it uses the rules only.

Status: export and runner steps are documented, and the integration code is in `advisor.py`. On-board latency of the Llama path is still to be measured.
