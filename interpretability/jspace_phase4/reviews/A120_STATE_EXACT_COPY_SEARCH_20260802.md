# A120 historical state exact-copy search

**ENGINEERING / GOVERNANCE EVIDENCE — NO SCIENTIFIC RESULT — NO RECOVERY
CLAIM — INDEPENDENT REVIEW AND PI DECISION STILL REQUIRED**

Search completed: 2026-08-02 10:10 UTC. Target:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/
metrics/qwen36-27b-multilens-functional-gate/functional_gate/
p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1/state.json
expected SHA-256 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8
```

No exact bytes were found and no file was restored. The missing state contains
irrecoverable timing and peak-allocation fields, so this negative search does
not authorize reconstruction, rerunning to manufacture substitute bytes, or
editing the append-only registry.

## Live Google Drive revision inventory

The read-only Drive v3 API was queried with the VM's existing short-lived
Drive authorization. The access token was kept in memory and was neither
printed nor persisted.

- The exact target evidence-directory name resolved to one live folder.
- Listing that folder returned thirteen live children. They are the same
  thirteen files visible through the mount; neither `state.json` nor
  `capacity_reconstructions_a120.pt` has a cloud file ID in that folder.
- An account-wide exact-name query returned sixty `state.json` files,
  including one irrelevant trashed object. None belongs to the target folder.
- An account-wide paginated `name contains 'state'` query returned 1,385
  objects. No name contained `a120` or the functional-gate evidence ID. The
  only object above 40 MB was the distinct registered A250--A500
  `state.json` (107,170,809 bytes).
- Revision listings covered 1,246 revisions across the sixty exact-name
  objects. Every revision above 50 MB belongs to that distinct A250--A500
  file. The API exposes no target file ID against which a target revision can
  be requested.

The cloud upload service continued to return `403 userRateLimitExceeded`
during this audit. That affects pending writes but does not create a hidden
revision surface for a file with no cloud ID.

## Preserved pre-incident DriveFS metadata

The quarantined profile was opened read-only with SQLite immutable mode. No
database or cache file was changed.

| database | bytes | SHA-256 |
|---|---:|---|
| `metadata_sqlite_db` | 54,419,456 | `e2c7a08b056dc14628630629c145856449be5d639edd04a1ae1511149874fe73` |
| `mirror_metadata_sqlite.db` | 679,936 | `4c1eab33f4efd124c9a2d343e3985c371685d8b571c08cfa22fcf2b8f6545d3e` |
| `mirror_sqlite.db` | 81,920 | `19b894c125d9b8352cd152303f87c300c04ce905067dd529ad4f041fa3d342bd` |

The primary metadata database contains the exact target folder and the same
thirteen children returned by the cloud API. Its `items` table contains sixty
objects titled `state.json`, but none is a child of the target folder. Its
`deleted_items` table is empty. Forty-one pending-operation protobufs were
searched for the target folder ID, `state.json`, and
`capacity_reconstructions_a120.pt`; the only state-name hit concerns an
unrelated mutable Qwen `checkpoint_state.json` temporary. The mirror database
contains no target-folder row, state row, pending upload, queued upload, or
pending delete.

The old profile's recreatable content cache was necessarily evicted after the
documented 130-GiB local-cache exhaustion incident. The metadata proves that
the target file did not have a recoverable cloud object or pending mirror
record at quarantine time; it cannot prove what bytes may once have existed
only in an already-evicted anonymous content chunk.

## Remaining legal resolution

The separate exact-hash-gated capacity reconstruction remains permitted after
A1000 releases the GPU. For `state.json`, the available VM files, mounted
Drive tree, live cloud/trash listing, cloud revision surface, and preserved
pre-incident metadata are exhausted. A newly discovered exact backup may
still be accepted only if its complete SHA-256 matches the registered value.
Otherwise an independent reviewer and the PI must select the append-only
resolution described in
`A120_FUNCTIONAL_DURABILITY_RECOVERY_INSTRUCTIONS.md`. The implementation
agent has not selected or approved that resolution.
