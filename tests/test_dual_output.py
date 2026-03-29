import unittest

import pandas as pd

from job_recommender.dual_output import (
    dual_search,
    InvalidRIASECCode,
    normalize_riasec_code,
    riasec_code_match,
    riasec_knn_search,
)


class TestRIASECValidation(unittest.TestCase):
    def test_normalize_accepts_letters_and_separators(self) -> None:
        self.assertEqual(normalize_riasec_code("r-i-a"), "RIA")
        self.assertEqual(normalize_riasec_code(" S E C "), "SEC")

    def test_normalize_rejects_invalid_letters(self) -> None:
        with self.assertRaises(InvalidRIASECCode):
            normalize_riasec_code("RXA")

    def test_normalize_rejects_duplicates(self) -> None:
        with self.assertRaises(InvalidRIASECCode):
            normalize_riasec_code("RRI")


class TestRIASECMatching(unittest.TestCase):
    def test_code_matching_prefers_exact_sequence(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "First Personality": "Realistic",
                    "Second Personality": "Investigative",
                    "Third Personality": "Artistic",
                    "Occupation": "Exact Match",
                    "Job Family": "Family A",
                },
                {
                    "First Personality": "Investigative",
                    "Second Personality": "Realistic",
                    "Third Personality": "Artistic",
                    "Occupation": "Partial Match",
                    "Job Family": "Family B",
                },
                {
                    "First Personality": "Social",
                    "Second Personality": "-",
                    "Third Personality": "-",
                    "Occupation": "No Match",
                    "Job Family": "Family C",
                },
            ]
        )
        out = riasec_code_match(df, "RIA", top_k=10)
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["Occupation"], "Exact Match")
        self.assertNotEqual(out.iloc[0]["match_score"], 0)

    def test_riasec_knn_prefers_exact_sequence(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "First Personality": "Realistic",
                    "Second Personality": "Investigative",
                    "Third Personality": "Artistic",
                    "Occupation": "Exact Match",
                    "Job Family": "Family A",
                },
                {
                    "First Personality": "Investigative",
                    "Second Personality": "Realistic",
                    "Third Personality": "Artistic",
                    "Occupation": "Swapped Match",
                    "Job Family": "Family B",
                },
            ]
        )
        out = riasec_knn_search(df, "RIA", top_k=2)
        self.assertEqual(out.iloc[0]["Occupation"], "Exact Match")


class TestDualSearchCompare(unittest.TestCase):
    def test_dual_search_can_return_multiple_text_methods(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "First Personality": "Realistic",
                    "Second Personality": "Investigative",
                    "Third Personality": "Artistic",
                    "Occupation": "Data Scientist",
                    "Job Family": "Computer",
                },
                {
                    "First Personality": "Social",
                    "Second Personality": "Enterprising",
                    "Third Personality": "Conventional",
                    "Occupation": "Sales Manager",
                    "Job Family": "Sales",
                },
            ]
        )
        result = dual_search(
            df,
            riasec_code="RIA",
            text_query="data modeling",
            top_k=2,
            text_methods=["tfidf", "count"],
            include_riasec_knn=True,
        )
        self.assertIn("tfidf", result.text_results_by_method)
        self.assertIn("count", result.text_results_by_method)
        self.assertIsNotNone(result.riasec_knn_results)


if __name__ == "__main__":
    unittest.main()
