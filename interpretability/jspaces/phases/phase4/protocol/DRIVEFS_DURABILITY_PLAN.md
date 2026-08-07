# Phase 4 DriveFS durability plan

**Binding engineering plan; it does not waive any missing artifact or
scientific freeze gate.**

DriveFS is a delivery mirror, not the only copy of a live scientific result.
The whole-registry clean criterion remains exact existence and SHA-256 for
every output of every live event. A known-deficit label explains a failure;
it never converts failure into success.

## Write path

1. Model producers write restart state and immutable outputs to local NVMe
   first, using a same-directory temporary file and atomic rename.
2. A completed output is hashed locally, copied to its final Drive path via a
   temporary sibling, hashed again, and atomically renamed only after the
   hashes agree. Registry creation happens after that verification.
3. Every newly registered event is immediately copied to the
   content-addressed local backup root
   `$JSPACE4_LOCAL_WORK/postfit_registered_backups/<evidence-id>/`, with a
   manifest recording registered path, backup path, bytes, and SHA-256.
4. Mutable recovery checkpoints live in explicitly named recovery
   directories and have a checkpoint/header pair. They are not evidence
   milestones. Immutable registered results never share their lifecycle.
5. Do not run broad Drive reads while a multi-gigabyte checkpoint mirror is
   in progress. A FUSE request stall is an operational stop signal, not
   permission to skip verification or restart the producer.

## Two-pass release audit

After model writers have stopped and Drive has quiesced:

```bash
python -m jspace_phase4.durability \
  --known-deficits interpretability/jspaces/phases/phase4/protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json \
  --pass-label local-plus-mounted-drive \
  --output /content/phase4_durability_pass1.json
```

The second pass must occur after an independent materialization boundary: a
Drive remount or a new VM that has not inherited the first pass's content
cache. It uses the same registry bytes and writes a different local report:

```bash
python -m jspace_phase4.durability \
  --known-deficits interpretability/jspaces/phases/phase4/protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json \
  --pass-label fresh-drive-materialization \
  --previous /content/phase4_durability_pass1.json \
  --output /content/phase4_durability_pass2.json
```

Release requires both reports to have `ok=true`, identical registry SHA-256,
identical reference sets, and no pass-to-pass hash/byte drift. Reports with
`only_known_deficits=true` remain red.

## Existing deficit and recovery rule

`KNOWN_DURABILITY_DEFICITS_PHASE4.json` names the two exact missing outputs
from the live A120--A250 functional event. Recover only bytes that rehash to
the registered values. Search content-addressed local backups, prior VM
archives, Drive version history, and explicitly preserved tarballs. Do not
reconstruct `state.json`, fabricate a tensor, edit the old event, or silently
withdraw it.

If exact bytes cannot be recovered, prepare an independently reviewed
append-only correction or supersession proposal that preserves the original
failure in history. Until either exact recovery or that reviewed governance
action is complete, whole-registry verification and Phase 4 freeze remain
blocked.

## Retention

- Keep local registered backups until two independent clean passes and a
  verified external/archive copy exist.
- Keep the final two snapshot JSON files, their SHA-256 values, the exact
  registry file, and the known-deficit manifest with the freeze package.
- Never delete the only local model/lens/output copy to make cache space while
  an owning job or an unresolved durability audit depends on it.
