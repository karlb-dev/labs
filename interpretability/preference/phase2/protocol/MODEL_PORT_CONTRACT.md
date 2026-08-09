# Model port contract (plan §50; D8-D10)

Four pinned cells (models.py; revisions frozen at prereg). Per model:
tokenizer/render audits (P2-5 artifact + rerun in-session), B-DEV strict
parse >= 0.98 per format, dev PCs expected, wrong branches 0, NC near
floor, folded carrier gap < 5.0 nats per pair (D10 ceiling; 0.10-nat
figure recorded as diagnostic), single-row replay exact, batched
generation equal, hook no-op, capture parity. Qwen: empty-think contract
(D8) via enable_thinking=False; non-empty/unclosed spans hard-fail.
Gemma: no-system fold with frozen separator; post-softcap logits are the
emission distribution; floors calibrated on Gemma's own NC rows. Failure
=> STOP_P(model), logged, other cells continue (E17).
