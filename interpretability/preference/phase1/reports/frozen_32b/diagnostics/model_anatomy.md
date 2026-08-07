# Model anatomy

What the bench resolved before any experiment ran. Every lab's hook
and readout uses these paths; if they look wrong, stop here.

| field | value |
|---|---|
| model | `allenai/Olmo-3.1-32B-Instruct` (revision: ac0587e4a7744a551c059d8cd17ba220bc940dae) |
| architecture | `Olmo3ForCausalLM` |
| parameters | 32,233,522,176 |
| decoder blocks | `model.layers` x 64 |
| final norm | `model.norm` (Olmo3RMSNorm) |
| unembedding | Linear, vocab 100,278 |
| d_model | 5120 |
| tied embeddings | False |
| logit softcap | none |

Depth convention used everywhere in this course: `streams[k]` is the
**pre-norm residual stream after k blocks**; k=0 is the embedding
output and k=64 is the input to the final norm.
