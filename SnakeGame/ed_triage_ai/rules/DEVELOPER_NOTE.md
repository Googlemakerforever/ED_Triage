# Safety Guardrail Developer Note

## What changed
- Added [`safety_guardrails.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/rules/safety_guardrails.py) as the mandatory deterministic safety layer.
- This layer now performs input validation, complaint normalization, negation-aware category detection, hard overrides, uncertainty escalation, audit generation, and lower-acuity default logic before any model inference.
- [`clinical_rules.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/rules/clinical_rules.py) is now a compatibility wrapper so existing imports continue to work.

## Where overrides run
- `TriagePredictor.predict()` calls `apply_clinical_rules()` before any model scoring.
- If the guardrail layer returns a result, that result is final.
- The model is only used when the safety layer does not fire.

## How to add future red-flag rules safely
- Add aliases to `CATEGORY_ALIASES` instead of hard-coding strings inline.
- Add new deterministic logic in `_level_1_overrides()`, `_level_2_overrides()`, `_uncertainty_escalation()`, or `_default_logic()` depending on risk.
- Add a stable internal rule ID such as `L2_NEW_RULE` so audits remain traceable.
- Add a test case in [`test_safety_guardrails.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/tests/test_safety_guardrails.py) for every new high-risk rule.

## How uncertainty escalation works
- If multiple medium-risk signals coexist, the complaint improves but still contains recent severe symptoms, or dangerous keywords remain after partial parsing, the system escalates instead of falling through to a lower-acuity path.
- The uncertainty path returns `source="uncertainty_escalation"` and preserves the matched rule IDs and audit metadata.
