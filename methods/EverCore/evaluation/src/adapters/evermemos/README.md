# EverMemOS Evaluation

The shared evaluation CLI runs this adapter from `methods/EverCore`:

```powershell
python -m evaluation.cli --dataset locomo --system evermemos
```

System settings: `evaluation/config/systems/evermemos.yaml`.
Pipeline options and outputs: [evaluation guide](../../../README.md).

## Stages

| Module | Operation |
| --- | --- |
| `stage1_memcells_extraction.py` | Extract memory cells, episodes, and atomic facts |
| `stage2_index_building.py` | Build vector and BM25 indexes |
| `stage3_memory_retrivel.py` | Retrieve evidence |
| `stage4_response.py` | Generate answers |
| `stage5_eval.py` | Evaluate answers |

`config.py` defines `ExperimentConfig`, including model endpoints, extraction
switches, retrieval depth, and concurrency.
`retrieval_mode` selects `agentic` or `lightweight` retrieval.
Lightweight retrieval supports `bm25_only`, `hybrid`, and `emb_only`;
agentic retrieval uses sufficiency checks and query refinement.

## Artifacts

Memory cells, BM25 indexes, and vectors are stored under the run directory,
alongside the shared search, answer, and evaluation outputs.
