import unittest
from typing import Tuple, Optional

# Import the functions we want to test
from wikipedia_retrieval import parse_financial_info


class TestFinancialInfoParsing(unittest.TestCase):
    def assert_parse_result(
        self,
        input_string: str,
        expected_output: Tuple[
            Optional[Tuple[float, float, float]], str, Optional[str]
        ],
    ):
        result = parse_financial_info(input_string)
        self.assertEqual(result, expected_output, f"Failed for input: {input_string}")

    def test_single_value(self):
        self.assert_parse_result(
            "$10 million",
            (
                (10000000.0, 10000000.0, 10000000.0),
                "USD Min: 10000000.00, Max: 10000000.00, Avg: 10000000.00",
                "USD",
            ),
        )

    def test_range(self):
        self.assert_parse_result(
            "$10-15 million",
            (
                (10000000.0, 15000000.0, 12500000.0),
                "USD Min: 10000000.00, Max: 15000000.00, Avg: 12500000.00",
                "USD",
            ),
        )

    def test_multiple_currencies(self):
        self.assert_parse_result(
            "DM$20 million(USD$12.1 million)",
            (
                (20000000.0, 20000000.0, 20000000.0),
                "DEM Min: 20000000.00, Max: 20000000.00, Avg: 20000000.00",
                "DEM",
            ),
        )

    def test_currency_with_space(self):
        self.assert_parse_result(
            "R$ 28 million (US$ 14.1 million)",
            (
                (28000000.0, 28000000.0, 28000000.0),
                "BRL Min: 28000000.00, Max: 28000000.00, Avg: 28000000.00",
                "BRL",
            ),
        )

    def test_admissions(self):
        self.assert_parse_result(
            "717,000 admissions (Germany)",
            (None, "Admissions data, not financial", None),
        )

    def test_no_currency(self):
        self.assert_parse_result(
            "10 million viewers",
            (None, "No recognized currency in: 10 million viewers", None),
        )

    def test_multiple_values_same_currency(self):
        self.assert_parse_result(
            "$10 million domestic and $5 million international",
            (
                (5000000.0, 10000000.0, 7500000.0),
                "USD Min: 5000000.00, Max: 10000000.00, Avg: 7500000.00",
                "USD",
            ),
        )

    def test_non_standard_currency(self):
        self.assert_parse_result(
            "SG$4,500,000",
            (
                (4500000.0, 4500000.0, 4500000.0),
                "SGD Min: 4500000.00, Max: 4500000.00, Avg: 4500000.00",
                "SGD",
            ),
        )

    def test_mixed_scales(self):
        self.assert_parse_result(
            "$1.5 billion to $2,000 million",
            (
                (1500000000.0, 2000000000.0, 1750000000.0),
                "USD Min: 1500000000.00, Max: 2000000000.00, Avg: 1750000000.00",
                "USD",
            ),
        )

    def test_invalid_input(self):
        self.assert_parse_result(
            "Not a valid financial string",
            (None, "No recognized currency in: not a valid financial string", None),
        )

    def test_no_value(self):
        self.assert_parse_result(
            "No value", (None, "No recognized currency in: no value", None)
        )

    def test_prob_case_1(self):
        self.assert_parse_result(
            "₺ 3,000,000[2]",
            (
                (3000000.0, 3000000.0, 3000000.0),
                "TRY Min: 3000000.00, Max: 3000000.00, Avg: 3000000.00",
                "TRY",
            ),
        )

    def test_prob_case_2(self):
        self.assert_parse_result(
            "50,000,000 NOK	",
            (
                (50000000.0, 50000000.0, 50000000.0),
                "NOK Min: 50000000.00, Max: 50000000.00, Avg: 50000000.00",
                "NOK",
            ),
        )

    def test_prob_case_3(self):
        self.assert_parse_result(
            "₹ 1.5 billion",
            (
                (1500000000.0, 1500000000.0, 1500000000.0),
                "INR Min: 1500000000.00, Max: 1500000000.00, Avg: 1500000000.00",
                "INR",
            ),
        )

    def test_prob_case_4(self):
        self.assert_parse_result(
            "€ 1.5 billion",
            (
                (1500000000.0, 1500000000.0, 1500000000.0),
                "EUR Min: 1500000000.00, Max: 1500000000.00, Avg: 1500000000.00",
                "EUR",
            ),
        )

    def test_prob_case_5(self):
        self.assert_parse_result(
            "127 million Yuan",
            (
                (127_000_000.0, 127_000_000.0, 127_000_000.0),
                "CNY Min: 127000000.00, Max: 127000000.00, Avg: 127000000.00",
                "CNY",
            ),
        )

    def test_prob_case_6(self):
        self.assert_parse_result(
            "$519–520.9 million",
            (
                (519_000_000.0, 520_900_000.0, 519_950_000.0),
                "USD Min: 519000000.00, Max: 520900000.00, Avg: 519950000.00",
                "USD",
            ),
        )

    def test_prob_case_7(self):
        self.assert_parse_result(
            "$14–15 million[2]	",
            (
                (14_000_000.0, 15_000_000.0, 14_500_000.0),
                "USD Min: 14000000.00, Max: 15000000.00, Avg: 14500000.00",
                "USD",
            ),
        )

    def test_prob_case_8(self):
        self.assert_parse_result(
            "$14–15 millions[2]	",
            (
                (14_000_000.0, 15_000_000.0, 14_500_000.0),
                "USD Min: 14000000.00, Max: 15000000.00, Avg: 14500000.00",
                "USD",
            ),
        )


class TestCurrencyDetection(unittest.TestCase):
    def test_detect_currencies(self):
        # self.assertEqual(detect_currencies("$10 million"), "USD")
        # self.assertEqual(detect_currencies("£10 million"), "GBP")
        # self.assertEqual(detect_currencies("10 million euros"), "EUR")
        # self.assertEqual(detect_currencies("DM 10 million"), "DEM")
        # self.assertEqual(detect_currencies("10 million yen"), "JPY")
        # self.assertEqual(detect_currencies("R$10 million"), "BRL")
        # self.assertEqual(detect_currencies("10 million kr"), "SEK")
        # self.assertEqual(detect_currencies("SG$10 million"), "SGD")
        # self.assertEqual(detect_currencies("10 million canadian dollar"), "CAD")
        # self.assertEqual(detect_currencies("10 million australian dollar"), "AUD")
        # self.assertEqual(detect_currencies("10 million rupees"), "INR")
        # self.assertEqual(detect_currencies("10 million yuan"), "CNY")
        # self.assertEqual(detect_currencies("10 million won"), "KRW")
        # self.assertEqual(detect_currencies("10 million rubles"), "RUB")
        # self.assertEqual(detect_currencies("$ 10 million"), "USD")
        # self.assertEqual(detect_currencies("€ 10 million"), "EUR")
        # self.assertEqual(detect_currencies("No currency here"), None)
        pass


if __name__ == "__main__":
    unittest.main()
