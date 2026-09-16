cat > README.md << 'EOF'
# JAX Transformer From Scratch

A decoder-only (GPT-style) transformer implemented from raw JAX primitives,
scaled dot-product attention, multi-head attention, causal masking, and the
training loop are written by hand rather than imported from a transformer
library. The model is trained on real accelerator hardware and extended to
data-parallel training across multiple devices.

The goal is to understand and demonstrate the layer of the stack beneath the
usual model-consumption workflow: how attention is actually computed, how JAX's
function transformations (`grad`, `jit`, `vmap`, `pmap`) compose, and what
distributed training mechanically involves.

## Status

- [ ] Phase 1 - JAX foundations
- [ ] Phase 2 - Attention and transformer blocks
- [ ] Phase 3 - Full model and tokenization
- [ ] Phase 4 - Training loop (local)
- [ ] Phase 5 - Accelerator training
- [ ] Phase 6 - Distributed training
- [ ] Phase 7 - CUDA kernel (stretch)
- [ ] Phase 8 - Paper reproduction (stretch)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Structure

- `src/` - model, training loop, parallel training step
- `tests/` - correctness tests
- `docs/` - design decisions, training log, distributed-training writeup
- `notebooks/` - Colab/Kaggle training notebooks