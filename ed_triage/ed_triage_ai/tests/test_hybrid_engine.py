import unittest

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.triage.hybrid_engine import HybridTriageEngine
from ed_triage_ai.triage.normalize_complaint import extract_critical_flags, normalize_complaint
from ed_triage_ai.triage.predict_acuity_ml import (
    build_runtime_features_for_binary_model,
    score_high_acuity_binary,
)


class FailingExtractor:
    def extract(self, complaint: str):
        raise RuntimeError("provider failed")


class LowConfidenceDangerousExtractor:
    def extract(self, complaint: str):
        return {
            "features": {
                "chest_pain": True,
                "shortness_of_breath": False,
                "stroke_like_symptoms": False,
                "head_injury": False,
                "loss_of_consciousness": False,
                "vomiting": False,
                "confusion": False,
                "amnesia": False,
                "possible_fracture": False,
                "open_injury": False,
                "deformity": False,
                "neurovascular_compromise": False,
                "dangerous_mechanism": False,
                "severe_bleeding": False,
                "severe_abdominal_pain": False,
                "pregnancy_related": False,
                "suicidal_risk": False,
                "homicidal_risk": False,
                "overdose_or_ingestion": False,
                "sepsis_concern": False,
                "altered_mental_status": False,
                "airway_compromise": False,
                "active_seizure": False,
                "severe_allergic_reaction": False,
                "minor_injury": False,
                "low_resource_case": False,
            },
            "context": {
                "negations": [],
                "temporal_modifiers": [],
                "confidence": 0.2,
                "ambiguity_flags": ["uncertain_text"],
                "extractor": "genai",
            },
        }


class HybridEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = TriagePredictor()

    def test_fallback_when_genai_fails(self):
        engine = HybridTriageEngine(self.predictor.model, feature_extractor=FailingExtractor())
        result = engine.run(
            {
                "age": 55,
                "sex": "Female",
                "heart_rate": 86,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 37.0,
                "systolic_bp": 126,
                "diastolic_bp": 78,
                "pain_score": 3,
                "chief_complaint": "chest pain while resting",
            }
        )
        self.assertEqual(result["source"], "hard_override")
        self.assertEqual(result["audit"]["extracted_features"]["context"]["extractor"], "keyword_fallback")

    def test_low_confidence_dangerous_text_applies_floor(self):
        engine = HybridTriageEngine(self.predictor.model, feature_extractor=LowConfidenceDangerousExtractor())
        result = engine.run(
            {
                "age": 48,
                "sex": "Male",
                "heart_rate": 78,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 122,
                "diastolic_bp": 76,
                "pain_score": 1,
                "chief_complaint": "unclear chest symptoms",
            }
        )
        self.assertIn(result["source"], {"uncertainty_escalation", "hard_override"})
        self.assertLessEqual(result["level"], 2)

    def test_gray_zone_case_uses_ml(self):
        result = self.predictor.predict(
            {
                "age": 29,
                "sex": "Female",
                "heart_rate": 82,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 2,
                "chief_complaint": "lightheaded when standing earlier today",
            }
        )
        self.assertEqual(result.prediction_source, "ml_prediction")
        self.assertTrue(result.audit["ml_probabilities"])

    def test_no_downgrade_over_open_injury(self):
        result = self.predictor.predict(
            {
                "age": 34,
                "sex": "Male",
                "heart_rate": 96,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 126,
                "diastolic_bp": 80,
                "pain_score": 6,
                "chief_complaint": "open fracture after motorcycle crash",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.prediction_source, "hard_override")
        self.assertIn("L2_OPEN_FRACTURE", result.matched_rules)

    def test_regression_head_injury_with_transient_loc_and_normal_vitals(self):
        result = self.predictor.predict(
            {
                "age": 54,
                "sex": "Male",
                "heart_rate": 92,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "pain_score": 3,
                "chief_complaint": "fell down stairs, hit head, briefly passed out, now feels okay but has mild headache",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.prediction_source, "hard_override")
        self.assertIn("L2_SEMANTIC_HEAD_INJURY", result.matched_rules)

    def test_semantic_loc_collapsed_briefly_and_came_to(self):
        result = self.predictor.predict(
            {
                "age": 52,
                "sex": "Male",
                "heart_rate": 88,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 126,
                "diastolic_bp": 78,
                "pain_score": 2,
                "chief_complaint": "collapsed briefly and then came to, now feels okay",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_loc"])

    def test_semantic_loc_lost_awareness_after_hitting_head(self):
        result = self.predictor.predict(
            {
                "age": 33,
                "sex": "Female",
                "heart_rate": 90,
                "respiratory_rate": 18,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 3,
                "chief_complaint": "lost awareness for a moment after hitting head",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_head_injury_red_flags"])

    def test_semantic_loc_went_down_for_seconds_after_fall(self):
        result = self.predictor.predict(
            {
                "age": 45,
                "sex": "Male",
                "heart_rate": 94,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 80,
                "pain_score": 2,
                "chief_complaint": "went down for a few seconds after the fall but now okay",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_loc"])

    def test_semantic_stroke_speech_sounded_off(self):
        result = self.predictor.predict(
            {
                "age": 61,
                "sex": "Female",
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 128,
                "diastolic_bp": 80,
                "pain_score": 1,
                "chief_complaint": "speech sounded off and right side felt strange",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_stroke"])

    def test_semantic_stroke_could_not_get_words_out(self):
        result = self.predictor.predict(
            {
                "age": 58,
                "sex": "Male",
                "heart_rate": 82,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 122,
                "diastolic_bp": 76,
                "pain_score": 1,
                "chief_complaint": "couldn't get words out for a minute but now better",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_stroke"])

    def test_no_loc_negation_does_not_trigger(self):
        flags = extract_critical_flags(normalize_complaint("head injury but no loss of consciousness"))
        self.assertFalse(flags["semantic_loc"]["flag"])

    def test_denies_chest_pain_medication_refill_remains_low(self):
        result = self.predictor.predict(
            {
                "age": 44,
                "sex": "Female",
                "heart_rate": 76,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 0,
                "chief_complaint": "denies chest pain, medication refill",
            }
        )
        self.assertIn(result.triage_level, {4, 5})
        self.assertFalse(result.audit["critical_flags"].get("semantic_stroke"))

    def test_binary_high_acuity_score_is_high_for_fall_with_loc(self):
        complaint = "fell down stairs, hit head, briefly passed out"
        runtime_features = build_runtime_features_for_binary_model(
            {
                "age": 54,
                "sex": "Male",
                "heart_rate": 96,
                "respiratory_rate": 20,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "pain_score": 6,
                "chief_complaint": complaint,
            },
            normalize_complaint(complaint),
        )
        binary_result = score_high_acuity_binary(runtime_features)
        self.assertGreater(binary_result["high_acuity_score"], 0.5)
        self.assertEqual(binary_result["high_acuity_pred"], 1)

    def test_binary_high_acuity_score_is_low_for_medication_refill(self):
        complaint = "medication refill"
        runtime_features = build_runtime_features_for_binary_model(
            {
                "age": 44,
                "sex": "Female",
                "heart_rate": 76,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 0,
                "chief_complaint": complaint,
            },
            normalize_complaint(complaint),
        )
        binary_result = score_high_acuity_binary(runtime_features)
        self.assertLess(binary_result["high_acuity_score"], 0.5)
        self.assertEqual(binary_result["high_acuity_pred"], 0)

    def test_stroke_semantic_headache_one_eye_blurry_with_arm_weakness(self):
        result = self.predictor.predict(
            {
                "age": 47,
                "sex": "Female",
                "heart_rate": 82,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 126,
                "diastolic_bp": 78,
                "pain_score": 6,
                "chief_complaint": "sudden headache and blurry vision in one eye, right arm feels weak",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_stroke"])

    def test_stroke_semantic_slight_slurring_and_left_sided_clumsiness(self):
        result = self.predictor.predict(
            {
                "age": 52,
                "sex": "Male",
                "heart_rate": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 128,
                "diastolic_bp": 80,
                "pain_score": 2,
                "chief_complaint": "slight slurring of speech and feels clumsy on left side",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(
            result.audit["critical_flags"]["semantic_stroke"]
            or result.audit["critical_flags"]["semantic_possible_stroke"]
        )

    def test_stroke_semantic_sudden_dizziness_and_trouble_walking_straight(self):
        result = self.predictor.predict(
            {
                "age": 63,
                "sex": "Male",
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.7,
                "systolic_bp": 132,
                "diastolic_bp": 82,
                "pain_score": 1,
                "chief_complaint": "sudden dizziness and trouble walking straight",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(
            result.audit["critical_flags"]["semantic_stroke"]
            or result.audit["critical_flags"]["semantic_possible_stroke"]
        )

    def test_classic_recurrent_migraine_pattern_not_level_two(self):
        result = self.predictor.predict(
            {
                "age": 31,
                "sex": "Female",
                "heart_rate": 78,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 74,
                "pain_score": 7,
                "chief_complaint": "severe throbbing headache with light sensitivity and nausea, similar to past migraines",
            }
        )
        self.assertNotEqual(result.triage_level, 2)
        self.assertFalse(result.audit["critical_flags"]["semantic_stroke"])

    def test_typical_migraine_aura_pattern_not_level_two(self):
        result = self.predictor.predict(
            {
                "age": 28,
                "sex": "Female",
                "heart_rate": 74,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.7,
                "systolic_bp": 116,
                "diastolic_bp": 72,
                "pain_score": 5,
                "chief_complaint": "zigzag lines in vision followed by headache, happens before migraines",
            }
        )
        self.assertNotEqual(result.triage_level, 2)
        self.assertFalse(result.audit["critical_flags"]["semantic_stroke"])

    def test_migraine_with_new_one_sided_weakness_is_level_two(self):
        result = self.predictor.predict(
            {
                "age": 39,
                "sex": "Female",
                "heart_rate": 80,
                "respiratory_rate": 15,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 6,
                "chief_complaint": "migraine with new one-sided weakness",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertTrue(result.audit["critical_flags"]["semantic_stroke"])


if __name__ == "__main__":
    unittest.main()
