# Lab 34 failure specimens

These are the rows that most shrink the favorite tool-state story. They are evidence, not clutter.

## failure_001: `surface_beats_probe` on `cal_008`

- required tool: `calendar`
- probe prediction: `file_search`
- surface prediction: `calendar`
- target shift at scale 1: `0.3125`
- random shift at scale 1: `0.3125`
- why it matters: The surface baseline explained the row better than the residual probe.

```text
Bug triage happens when?
```

## failure_002: `no_tool_surface_cue_false_positive` on `none_018`

- required tool: `none`
- probe prediction: `calculator`
- surface prediction: `calculator`
- target shift at scale 1: `-0.0625`
- random shift at scale 1: `0.1875`
- why it matters: No-tool rows with tool words are the main guard against intention language.

```text
17 * 23 appears on a sticker; answer sticker.
```

## failure_003: `no_tool_surface_cue_false_positive` on `none_016`

- required tool: `none`
- probe prediction: `file_search`
- surface prediction: `route_planner`
- target shift at scale 1: `0.125`
- random shift at scale 1: `-0.0625`
- why it matters: No-tool rows with tool words are the main guard against intention language.

```text
Route planner is a product name; answer product.
```

## failure_004: `no_tool_surface_cue_false_positive` on `none_022`

- required tool: `none`
- probe prediction: `file_search`
- surface prediction: `route_planner`
- target shift at scale 1: `0.4375`
- random shift at scale 1: `0.125`
- why it matters: No-tool rows with tool words are the main guard against intention language.

```text
A path A -> F is drawn as art; answer art.
```

## failure_005: `surface_beats_probe` on `cal_007`

- required tool: `calendar`
- probe prediction: `file_search`
- surface prediction: `calendar`
- target shift at scale 1: `0.0`
- random shift at scale 1: `0.0`
- why it matters: The surface baseline explained the row better than the residual probe.

```text
Design review slot?
```

## failure_006: `probe_tool_confusion` on `route_009`

- required tool: `route_planner`
- probe prediction: `file_search`
- surface prediction: `none`
- target shift at scale 1: `-0.0625`
- random shift at scale 1: `-0.375`
- why it matters: Which-tool decoding failed on a held-out task.

```text
A reaches D by which nodes?
```

## failure_007: `random_direction_matches_or_beats_target_direction` on `file_000`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.4375`
- random shift at scale 1: `0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Which synthetic document mentions cache invalidation?
Action letter:
```

## failure_008: `random_direction_matches_or_beats_target_direction` on `file_000`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.4375`
- random shift at scale 1: `0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Which synthetic document mentions cache invalidation?
Action letter:
```

## failure_009: `random_direction_matches_or_beats_target_direction` on `file_000`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.4375`
- random shift at scale 1: `0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Which synthetic document mentions cache invalidation?
Action letter:
```

## failure_010: `random_direction_matches_or_beats_target_direction` on `file_009`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5`
- random shift at scale 1: `-0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Stale user records after writes: name the document.
Action letter:
```

## failure_011: `random_direction_matches_or_beats_target_direction` on `file_009`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5`
- random shift at scale 1: `-0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Stale user records after writes: name the document.
Action letter:
```

## failure_012: `random_direction_matches_or_beats_target_direction` on `file_009`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5`
- random shift at scale 1: `-0.125`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Stale user records after writes: name the document.
Action letter:
```

## failure_013: `random_direction_matches_or_beats_target_direction` on `file_008`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5625`
- random shift at scale 1: `-0.0625`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Exporter CSV belongs to which document?
Action letter:
```

## failure_014: `random_direction_matches_or_beats_target_direction` on `file_008`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5625`
- random shift at scale 1: `-0.0625`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Exporter CSV belongs to which document?
Action letter:
```

## failure_015: `random_direction_matches_or_beats_target_direction` on `file_008`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.5625`
- random shift at scale 1: `-0.0625`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Exporter CSV belongs to which document?
Action letter:
```

## failure_016: `random_direction_matches_or_beats_target_direction` on `file_007`

- required tool: `file_search`
- probe prediction: `file_search`
- surface prediction: `file_search`
- target shift at scale 1: `-0.4375`
- random shift at scale 1: `0.0`
- why it matters: The causal letter-prompt test is not specific if random shifts as much as the target direction.

```text
Choose the best next action for this controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: Cache invalidation appears in which synthetic note?
Action letter:
```
