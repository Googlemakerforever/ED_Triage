import unittest

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.triage.hybrid_engine import HybridTriageEngine


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


if __name__ == "__main__":
    unittest.main()
