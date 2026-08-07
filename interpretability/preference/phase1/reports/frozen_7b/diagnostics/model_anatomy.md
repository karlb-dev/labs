# Model anatomy

What the bench resolved before any experiment ran. Every lab's hook
and readout uses these paths; if they look wrong, stop here.

| field | value |
|---|---|
| model | `allenai/Olmo-3-7B-Instruct` (revision: 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc) |
| architecture | `Olmo3ForCausalLM` |
| parameters | 7,298,011,136 |
| decoder blocks | `model.layers` x 32 |
| final norm | `model.norm` (Olmo3RMSNorm) |
| unembedding | Linear, vocab 100,278 |
| d_model | 4096 |
| tied embeddings | False |
| logit softcap | none |

Depth convention used everywhere in this course: `streams[k]` is the
**pre-norm residual stream after k blocks**; k=0 is the embedding
output and k=32 is the input to the final norm.
