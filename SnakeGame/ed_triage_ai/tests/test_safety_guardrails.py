import unittest

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.rules.safety_guardrails import evaluate_safety_guardrails


class SafetyGuardrailTests(unittest.TestCase):
    def test_life_threatening_instability(self):
        result = evaluate_safety_guardrails(
            {
                "age": 40,
                "sex": "Male",
                "heart_rate": 128,
                "respiratory_rate": 34,
                "oxygen_saturation": 82,
                "temperature": 36.8,
                "systolic_bp": 76,
                "diastolic_bp": 44,
                "pain_score": 2,
                "chief_complaint": "not breathing after collapse",
            }
        )
        self.assertEqual(result.triage_level, 1)
        self.assertEqual(result.source, "hard_override")
        self.assertIn("L1_APNEA", result.matched_rules)

    def test_chest_pain_with_normal_vitals_escalates(self):
        result = evaluate_safety_guardrails(
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
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.source, "hard_override")

    def test_neurologic_emergency_with_low_pain(self):
        result = evaluate_safety_guardrails(
            {
                "age": 64,
                "sex": "Male",
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.7,
                "systolic_bp": 132,
                "diastolic_bp": 82,
                "pain_score": 1,
                "chief_complaint": "new slurred speech and right sided weakness",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertIn("L2_STROKE_SIGNS", result.matched_rules)

    def test_head_injury_loc(self):
        result = evaluate_safety_guardrails(
            {
                "age": 22,
                "sex": "Male",
                "heart_rate": 88,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 74,
                "pain_score": 4,
                "chief_complaint": "hit head and briefly passed out after fall",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_shortness_of_breath_typo_still_escalates(self):
        result = evaluate_safety_guardrails(
            {
                "age": 31,
                "sex": "Female",
                "heart_rate": 102,
                "respiratory_rate": 24,
                "oxygen_saturation": 95,
                "temperature": 37.2,
                "systolic_bp": 122,
                "diastolic_bp": 76,
                "pain_score": 2,
                "chief_complaint": "shorness of breath since this morning",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_open_fracture(self):
        result = evaluate_safety_guardrails(
            {
                "age": 30,
                "sex": "Male",
                "heart_rate": 96,
                "respiratory_rate": 18,
                "oxygen_saturation": 99,
                "temperature": 36.9,
                "systolic_bp": 128,
                "diastolic_bp": 78,
                "pain_score": 6,
                "chief_complaint": "open fracture with bone exposed",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_closed_fracture(self):
        result = evaluate_safety_guardrails(
            {
                "age": 30,
                "sex": "Male",
                "heart_rate": 86,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 122,
                "diastolic_bp": 78,
                "pain_score": 7,
                "chief_complaint": "possible broken bone after ankle injury",
            }
        )
        self.assertEqual(result.triage_level, 3)

    def test_deformity_with_intact_circulation(self):
        result = evaluate_safety_guardrails(
            {
                "age": 44,
                "sex": "Female",
                "heart_rate": 90,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 72,
                "pain_score": 5,
                "chief_complaint": "arm deformity but warm hand and moving fingers",
            }
        )
        self.assertEqual(result.triage_level, 3)

    def test_neurovascular_compromise(self):
        result = evaluate_safety_guardrails(
            {
                "age": 37,
                "sex": "Male",
                "heart_rate": 94,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 126,
                "diastolic_bp": 80,
                "pain_score": 4,
                "chief_complaint": "leg injury with cold limb and absent distal pulse",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_dangerous_mechanism_with_benign_vitals(self):
        result = evaluate_safety_guardrails(
            {
                "age": 27,
                "sex": "Male",
                "heart_rate": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.7,
                "systolic_bp": 122,
                "diastolic_bp": 78,
                "pain_score": 2,
                "chief_complaint": "rollover crash but feels okay",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_minor_injury_lower_acuity(self):
        result = evaluate_safety_guardrails(
            {
                "age": 19,
                "sex": "Female",
                "heart_rate": 78,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 116,
                "diastolic_bp": 70,
                "pain_score": 3,
                "chief_complaint": "minor sprain after basketball",
            }
        )
        self.assertEqual(result.triage_level, 4)

    def test_pain_only_does_not_over_escalate(self):
        result = evaluate_safety_guardrails(
            {
                "age": 25,
                "sex": "Female",
                "heart_rate": 82,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 9,
                "chief_complaint": "ankle pain",
            }
        )
        self.assertNotIn(result.triage_level if result else 5, [1, 2])

    def test_negation_handling(self):
        result = evaluate_safety_guardrails(
            {
                "age": 41,
                "sex": "Male",
                "heart_rate": 76,
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 78,
                "pain_score": 0,
                "chief_complaint": "no chest pain denies shortness of breath medication refill",
            }
        )
        self.assertEqual(result.triage_level, 5)

    def test_ambiguity_escalation(self):
        result = evaluate_safety_guardrails(
            {
                "age": 34,
                "sex": "Female",
                "heart_rate": 84,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 74,
                "pain_score": 1,
                "chief_complaint": "briefly blacked out but feels better now",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.source, "uncertainty_escalation")

    def test_multiple_symptoms_highest_risk_wins(self):
        result = evaluate_safety_guardrails(
            {
                "age": 50,
                "sex": "Male",
                "heart_rate": 88,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 124,
                "diastolic_bp": 80,
                "pain_score": 2,
                "chief_complaint": "small bruise but also new slurred speech",
            }
        )
        self.assertEqual(result.triage_level, 2)

    def test_validation_error_for_impossible_values(self):
        result = evaluate_safety_guardrails(
            {
                "age": 25,
                "sex": "Female",
                "heart_rate": 500,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 3,
                "chief_complaint": "minor cut",
            }
        )
        self.assertEqual(result.source, "validation_error")

    def test_model_cannot_override_guardrail(self):
        predictor = TriagePredictor()
        result = predictor.predict(
            {
                "age": 32,
                "sex": "Male",
                "heart_rate": 92,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 126,
                "diastolic_bp": 78,
                "pain_score": 6,
                "chief_complaint": "open fracture after motorcycle crash",
            }
        )
        self.assertEqual(result.triage_level, 2)
        self.assertEqual(result.prediction_source, "hard_override")
        self.assertIn("L2_OPEN_FRACTURE", result.matched_rules)


if __name__ == "__main__":
    unittest.main()
