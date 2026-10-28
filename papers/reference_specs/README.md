# Reference specs (PRELIMINARY)

These StrategySpec-shaped JSON files are PRELIMINARY references derived mechanically
from `papers/answer_key.json`. They are not page-cited (every evidence object is
`page 0`, "reference spec (preliminary answer key); no quote") and fields the key
does not cover are set to "not stated in key" / "not in key".

They are to be replaced by the group's independently written specs.

Use: ONLY the interpretation scorer,
`research_engine/evaluation/interpretation.py` (`score_interpretation(run_dir,
reference_spec_path)`), which labels any reference whose path contains
"preliminary" as `preliminary: true`. Extraction and validation code must never
read these files.

Mapping: favored_group = key long_leg; disfavored_group = key short_leg;
expected_sign "favored_group_higher" when the key's long-short sign is positive.
The seasonality reference keeps the key's lag-12 wording, so an extraction that
chooses years 2-5 will show as a difference.
