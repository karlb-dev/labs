"""Incidental surface skins (plan §14: incidentals vary surface only).

32 fixed project ecosystems shared campaign-wide. Every skin fills the
same parameter vocabulary; templates never receive a parameter that
changes a tradeoff. Split assignment is positional and frozen at
authoring: B-ARB3 uses skins 0-15 (8 train / 4 validation / 4 holdout),
B-MECH uses skins 0-31 (16 train / 8 validation / 8 holdout), B-CANON
skins 0-7, B-PC skins 0-4, B-NC skins 0-7, B-SURF skins 0-7.
"""

from __future__ import annotations

from .schema import IncidentalSpec

_SKINS: list[dict[str, str]] = [
    # project, domain, module, src_fn, src_arg, suite, test_a, test_b,
    # datafile, service, table, feed
    dict(project="logmill", domain="log analytics", module="rotor",
         src_fn="merge", src_arg="items", suite="test_rotor.py",
         test_a="test_rollover", test_b="test_windowing",
         datafile="events.log", service="collector", table="log_entries",
         feed="syslog stream"),
    dict(project="tracknet", domain="route planning", module="pather",
         src_fn="score", src_arg="edges", suite="test_pather.py",
         test_a="test_detour", test_b="test_junction",
         datafile="routes.dat", service="planner", table="road_segments",
         feed="traffic feed"),
    dict(project="feedhub", domain="feed aggregation", module="fetcher",
         src_fn="dedupe", src_arg="entries", suite="test_fetcher.py",
         test_a="test_backoff", test_b="test_ordering",
         datafile="feeds.xml", service="aggregator", table="feed_items",
         feed="publisher feed"),
    dict(project="mailsort", domain="mail routing", module="router",
         src_fn="classify", src_arg="headers", suite="test_router.py",
         test_a="test_bounce", test_b="test_threading",
         datafile="inbox.mbox", service="sorter", table="messages",
         feed="smtp intake"),
    dict(project="shopfeed", domain="price tracking", module="pricer",
         src_fn="normalize", src_arg="rows", suite="test_pricer.py",
         test_a="test_currency", test_b="test_rounding",
         datafile="prices.csv", service="tracker", table="offers",
         feed="vendor feed"),
    dict(project="cachewarm", domain="cache priming", module="primer",
         src_fn="rank", src_arg="keys", suite="test_primer.py",
         test_a="test_eviction", test_b="test_prefetch",
         datafile="hotkeys.txt", service="warmer", table="cache_keys",
         feed="hit-rate feed"),
    dict(project="queuepilot", domain="job scheduling", module="dispatch",
         src_fn="assign", src_arg="jobs", suite="test_dispatch.py",
         test_a="test_retry", test_b="test_priority",
         datafile="queue.db", service="scheduler", table="job_runs",
         feed="worker heartbeat"),
    dict(project="tagforge", domain="asset tagging", module="labeler",
         src_fn="apply", src_arg="assets", suite="test_labeler.py",
         test_a="test_aliases", test_b="test_casing",
         datafile="assets.json", service="tagger", table="asset_tags",
         feed="upload feed"),
    dict(project="metricbay", domain="metric storage", module="ingest",
         src_fn="bucket", src_arg="points", suite="test_ingest.py",
         test_a="test_gap_fill", test_b="test_downsample",
         datafile="metrics.tsv", service="gateway", table="datapoints",
         feed="agent stream"),
    dict(project="snapvault", domain="snapshot backup", module="differ",
         src_fn="chunk", src_arg="blocks", suite="test_differ.py",
         test_a="test_dedup", test_b="test_manifest",
         datafile="snap.idx", service="archiver", table="snapshots",
         feed="volume scan"),
    dict(project="cronreef", domain="task automation", module="ticker",
         src_fn="expand", src_arg="specs", suite="test_ticker.py",
         test_a="test_skew", test_b="test_overlap",
         datafile="crontab.yml", service="runner", table="tick_events",
         feed="clock source"),
    dict(project="docknote", domain="release notes", module="drafter",
         src_fn="collect", src_arg="commits", suite="test_drafter.py",
         test_a="test_grouping", test_b="test_links",
         datafile="notes.md", service="publisher", table="note_entries",
         feed="commit log"),
    dict(project="parseline", domain="text extraction", module="scanner",
         src_fn="split", src_arg="lines", suite="test_scanner.py",
         test_a="test_quoting", test_b="test_escapes",
         datafile="corpus.txt", service="extractor", table="spans",
         feed="document drop"),
    dict(project="heapwatch", domain="memory profiling", module="sampler",
         src_fn="walk", src_arg="frames", suite="test_sampler.py",
         test_a="test_leak_flag", test_b="test_interning",
         datafile="heap.prof", service="profiler", table="allocations",
         feed="runtime probe"),
    dict(project="routeplan", domain="delivery planning", module="binder",
         src_fn="cluster", src_arg="stops", suite="test_binder.py",
         test_a="test_capacity", test_b="test_windows",
         datafile="stops.geojson", service="optimizer", table="route_stops",
         feed="order intake"),
    dict(project="syncbarn", domain="file synchronization", module="ledger",
         src_fn="diff", src_arg="trees", suite="test_ledger.py",
         test_a="test_renames", test_b="test_conflicts",
         datafile="state.db", service="syncer", table="file_states",
         feed="watcher events"),
    dict(project="filecrest", domain="document storage", module="indexer",
         src_fn="tokenize", src_arg="docs", suite="test_indexer.py",
         test_a="test_stemming", test_b="test_fields",
         datafile="docs.sqlite", service="librarian", table="doc_terms",
         feed="upload queue"),
    dict(project="wordbale", domain="translation memory", module="matcher",
         src_fn="align", src_arg="segments", suite="test_matcher.py",
         test_a="test_fuzzy", test_b="test_placeholders",
         datafile="memory.tmx", service="translator", table="segments",
         feed="editor saves"),
    dict(project="pixelfold", domain="image processing", module="resizer",
         src_fn="scale", src_arg="tiles", suite="test_resizer.py",
         test_a="test_aspect", test_b="test_formats",
         datafile="gallery.idx", service="renderer", table="derivatives",
         feed="upload burst"),
    dict(project="streamcut", domain="video clipping", module="slicer",
         src_fn="mark", src_arg="frames", suite="test_slicer.py",
         test_a="test_keyframes", test_b="test_audio_sync",
         datafile="edits.json", service="cutter", table="clip_marks",
         feed="ingest bucket"),
    dict(project="batchknot", domain="ETL orchestration", module="stager",
         src_fn="stage", src_arg="tables", suite="test_stager.py",
         test_a="test_ordering", test_b="test_rollback",
         datafile="stage.parquet", service="loader", table="stage_rows",
         feed="warehouse drop"),
    dict(project="indexloom", domain="search indexing", module="weaver",
         src_fn="shard", src_arg="terms", suite="test_weaver.py",
         test_a="test_merges", test_b="test_deletes",
         datafile="index.seg", service="searcher", table="postings",
         feed="crawler output"),
    dict(project="graphmint", domain="dependency graphing", module="walker",
         src_fn="visit", src_arg="nodes", suite="test_walker.py",
         test_a="test_cycles", test_b="test_orphans",
         datafile="graph.dot", service="resolver", table="graph_edges",
         feed="manifest scan"),
    dict(project="tokenmill", domain="API credential rotation", module="minter",
         src_fn="rotate", src_arg="grants", suite="test_minter.py",
         test_a="test_expiry", test_b="test_scopes",
         datafile="grants.db", service="issuer", table="tokens",
         feed="request log"),
    dict(project="schemaport", domain="schema migration", module="porter",
         src_fn="map", src_arg="columns", suite="test_porter.py",
         test_a="test_defaults", test_b="test_nullable",
         datafile="schema.sql", service="migrator", table="column_maps",
         feed="dump import"),
    dict(project="auditnest", domain="audit logging", module="notary",
         src_fn="seal", src_arg="records", suite="test_notary.py",
         test_a="test_chaining", test_b="test_redaction",
         datafile="audit.log", service="recorder", table="audit_rows",
         feed="event bus"),
    dict(project="deltapress", domain="data compression", module="packer",
         src_fn="encode", src_arg="chunks", suite="test_packer.py",
         test_a="test_ratios", test_b="test_integrity",
         datafile="delta.bin", service="compressor", table="chunk_stats",
         feed="archive queue"),
    dict(project="mergefield", domain="record deduplication", module="joiner",
         src_fn="link", src_arg="records", suite="test_joiner.py",
         test_a="test_survivorship", test_b="test_thresholds",
         datafile="entities.ndj", service="deduper", table="match_pairs",
         feed="crm export"),
    dict(project="sortgale", domain="ticket triage", module="triager",
         src_fn="route", src_arg="tickets", suite="test_triager.py",
         test_a="test_priorities", test_b="test_labels",
         datafile="tickets.json", service="triage", table="ticket_routes",
         feed="support inbox"),
    dict(project="packlane", domain="artifact packaging", module="bundler",
         src_fn="bundle", src_arg="targets", suite="test_bundler.py",
         test_a="test_signing", test_b="test_layers",
         datafile="bundle.lock", service="packer", table="artifacts",
         feed="build output"),
    dict(project="hashgrove", domain="content addressing", module="hasher",
         src_fn="digest", src_arg="blobs", suite="test_hasher.py",
         test_a="test_collisions", test_b="test_streaming",
         datafile="objects.pack", service="store", table="blob_refs",
         feed="object intake"),
    dict(project="notewheel", domain="meeting summaries", module="scribe",
         src_fn="outline", src_arg="notes", suite="test_scribe.py",
         test_a="test_speakers", test_b="test_actions",
         datafile="minutes.md", service="summarizer", table="note_items",
         feed="transcript drop"),
]

assert len(_SKINS) == 32
assert len({s["project"] for s in _SKINS}) == 32


def incidentals(n: int, *, splits: tuple[int, int, int]) -> tuple[IncidentalSpec, ...]:
    """First ``n`` skins with a (train, validation, holdout) split by
    position — frozen, never data-dependent."""
    n_tr, n_va, n_ho = splits
    assert n_tr + n_va + n_ho == n <= len(_SKINS)
    out = []
    for i in range(n):
        split = ("train" if i < n_tr
                 else "validation" if i < n_tr + n_va
                 else "holdout")
        out.append(IncidentalSpec(
            incidental_id=f"i{i:02d}", incidental_split=split,
            params=dict(_SKINS[i]),
        ))
    return tuple(out)


# 24 incidentals (12/6/6): raised from the plan's 16 by the power
# simulation (strict-choice power at the 0.10 SESOI was 0.45 at 16 and
# 0.87 at 24 under the E16 exact sign-flip + Holm-12 primary; plan §32
# commands the raise before freeze; addendum D permits raise-never-lower).
ARB3_INCIDENTALS = incidentals(24, splits=(12, 6, 6))
MECH_INCIDENTALS = incidentals(32, splits=(16, 8, 8))
CANON_INCIDENTALS = incidentals(8, splits=(8, 0, 0))
PC_INCIDENTALS = incidentals(5, splits=(5, 0, 0))
NC_INCIDENTALS = incidentals(8, splits=(8, 0, 0))
SURF_INCIDENTALS = incidentals(8, splits=(8, 0, 0))
DEV_INCIDENTALS = incidentals(4, splits=(4, 0, 0))
CONT_INCIDENTALS = incidentals(6, splits=(6, 0, 0))
