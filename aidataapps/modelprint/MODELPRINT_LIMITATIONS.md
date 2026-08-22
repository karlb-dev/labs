# ModelPrint Limitations

ModelPrint measures served-profile attribution under a frozen local experiment. It does not establish universal textual fingerprints.

- The four classes are particular repositories, revisions, templates, parsers, images, engine arguments, and decode policies. Revision or deployment changes can invalidate a classifier.
- Prompt, domain, carrier, length, language, reasoning policy, safety tuning, and decoding can dominate model identity. The lab measures these nuisance variables but cannot exhaust them.
- Closed-set accuracy says nothing about arbitrary unseen sources. OOD evaluation reduces forced attribution; it cannot guarantee novelty detection in the open world.
- Human controls are descriptive OOD cases, not training data for a human-versus-machine detector.
- Cross-likelihood requires local access to all candidate models and uses each served profile's tokenizer/template. Muse final-answer scoring excludes its hidden reasoning span.
- Learned projections can encode experimental artifacts. Family/source holdouts, name masking, permutation nulls, and nuisance-label comparisons limit but do not eliminate that risk.
- Very short responses may not contain enough evidence for a calibrated decision. The correct application behavior is `insufficient_text`, not a confident guess.
- Exact string collisions across profiles are intrinsically ambiguous for text-only attribution.
- SQL vector retrieval returns similar retained witnesses. It is neither a probability nor a proof of authorship.
- ANN is an engineering optimization with build-specific behavior. Every claim depends on recorded syntax, index metadata, query plan, recall, and decision agreement.
- Bootstrap intervals describe uncertainty under the sampled prompt groups; they do not cover all possible prompts or future distributions.
