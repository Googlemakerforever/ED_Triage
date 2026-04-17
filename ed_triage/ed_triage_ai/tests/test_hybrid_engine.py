import unittest

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.triage.hybrid_engine import HybridTriageEngine
from ed_triage_ai.triage.normalize_complaint import extract_critical_flags, normalize_complaint


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


if __name__ == "__main__":
    unittest.main()
