# Advisory module

Turns the detections for one board into a short recommendation for the operator.

- `advisor.py`: rule table per defect type (instant) and the Llama 3.2 1B path through the ExecuTorch runner
- `prompts/`: system prompt and the per-board message template
- `export_llama_executorch.md`: exporting Llama 3.2 1B, building the runner on the KV260 and connecting it to the station

Quick test without the board:

```bash
python3 advisor.py short 0.82 open_circuit 0.73
```
