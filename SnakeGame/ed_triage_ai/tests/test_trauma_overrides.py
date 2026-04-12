import unittest

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.rules.clinical_rules import apply_trauma_overrides


class TraumaOverrideTests(unittest.TestCase):
    def test_level_1_instability_override(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 118,
                "respiratory_rate": 20,
                "oxygen_saturation": 82,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 4,
                "chief_complaint": "major trauma after rollover crash",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 1)
        self.assertEqual(result.source, "trauma_override")

    def test_level_2_open_fracture_and_neurovascular_compromise(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 98,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "pain_score": 6,
                "chief_complaint": "open fracture with bone exposed and absent distal pulse after injury",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 2)

    def test_level_3_closed_fracture_features(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 122,
                "diastolic_bp": 76,
                "pain_score": 8,
                "chief_complaint": "possible broken bone with deformity and unable to bear weight after fall",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 3)

    def test_level_4_minor_stable_injury(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 78,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.7,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 3,
                "chief_complaint": "minor sprain ambulatory and moving limb",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 4)

    def test_level_5_trivial_injury(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 76,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 116,
                "diastolic_bp": 72,
                "pain_score": 1,
                "chief_complaint": "superficial abrasion healed injury recheck only no deformity no functional loss",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 5)

    def test_dangerous_mechanism_escalates_with_normal_vitals(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 78,
                "pain_score": 2,
                "chief_complaint": "high-speed collision with normal vitals",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.triage_level, 2)

    def test_pain_only_case_does_not_over_escalate(self):
        result = apply_trauma_overrides(
            {
                "heart_rate": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 8,
                "chief_complaint": "severe ankle pain",
            }
        )
        self.assertIsNone(result)

    def test_predictor_cannot_overwrite_trauma_override(self):
        predictor = TriagePredictor()
        result = predictor.predict(
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
                "chief_complaint": "open fracture with bone exposed after motorcycle crash",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.prediction_source, "trauma_override")
        self.assertTrue(result.override_triggered)


if __name__ == "__main__":
    unittest.main()
