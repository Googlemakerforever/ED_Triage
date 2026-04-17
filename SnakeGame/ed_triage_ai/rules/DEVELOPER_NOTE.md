# Hybrid Triage Developer Note

## What changed
- Added a new modular hybrid engine centered in [`hybrid_engine.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/hybrid_engine.py).
- The engine now runs this order: validation -> normalization -> structured feature extraction -> hard overrides -> uncertainty floor -> ML prediction -> no-downgrade combiner -> audited output.
- [`predict.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/models/predict.py) now uses that hybrid engine rather than letting the model decide first.

## Where GenAI is used
- GenAI is only used in [`extract_structured_features.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/extract_structured_features.py).
- It extracts structured complaint flags and context metadata.
- It does not return the final triage level.
- If GenAI fails or is unavailable, [`fallback_keyword_extractor.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/fallback_keyword_extractor.py) takes over deterministically.

## Where deterministic overrides run
- Hard overrides live in [`apply_hard_overrides.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/apply_hard_overrides.py).
- Uncertainty escalation and safety floors live in [`apply_uncertainty_escalation.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/apply_uncertainty_escalation.py).
- These run before ML and can block low-acuity outputs.

## Where ML is used
- ML runs only for gray-zone cases in [`predict_acuity_ml.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/predict_acuity_ml.py).
- It uses the existing trained tabular pipeline after safety screening.

## How no-downgrade protection works
- [`combine_with_safety_floor.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/combine_with_safety_floor.py) prevents ML from lowering a higher-acuity deterministic rule or uncertainty floor.
- Hard overrides always win.
- If ML predicts a higher-acuity level than the floor, ML can escalate upward, but not downward.

## How to extend safely
- Add new aliases or synonyms in [`keyword_sets.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/triage/keyword_sets.py).
- Add new deterministic rule IDs in the override or uncertainty modules.
- Add a test case in [`test_hybrid_engine.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/tests/test_hybrid_engine.py) or [`test_safety_guardrails.py`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/ed_triage_ai/tests/test_safety_guardrails.py) for every new high-risk rule.
