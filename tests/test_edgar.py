"""SEC EDGAR extraction and labelling, on small hand-made fact documents.

Run: python -m unittest discover -s tests -t .
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import settings
from src.data_collection import edgar_quarterly as eq
from src.data_collection.sec_edgar import EdgarClient, EdgarConfigError
from src.ml.edgar_dataset import label_company, nearest_quarter_end


def fact(start, end, val, filed, form="10-Q"):
    item = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        item["start"] = start
    return item


def facts_doc(concepts):
    return {"facts": {"us-gaap": {
        name: {"units": {"USD": items}} for name, items in concepts.items()
    }}}


# One fiscal year (calendar 2012) shaped like Amazon's real 2012 facts.
REVENUE_2012 = [
    fact("2012-01-01", "2012-03-31", 13.0, "2012-04-27"),
    fact("2012-04-01", "2012-06-30", 12.0, "2012-07-27"),
    fact("2012-07-01", "2012-09-30", 14.0, "2012-10-26"),
    fact("2012-01-01", "2012-09-30", 39.0, "2012-10-26"),
    fact("2012-01-01", "2012-12-31", 60.0, "2013-01-30", "10-K"),
    # The 10-K also tags a wrong (segment-level) 3-month Q4 figure ...
    fact("2012-10-01", "2012-12-31", 17.0, "2013-01-30", "10-K"),
    # ... and restates Q1 as a comparative a year later.
    fact("2012-01-01", "2012-03-31", 99.0, "2014-01-31", "10-K"),
    fact("2013-01-01", "2013-03-31", 16.0, "2013-04-26"),
]


class ExtractionTests(unittest.TestCase):
    def test_q4_derived_and_first_filed_kept(self):
        doc = facts_doc({"SalesRevenueNet": REVENUE_2012})["facts"]["us-gaap"]
        quarters = eq.quarterly_flow(doc, "SalesRevenueNet")
        values = {k.strftime("%Y-%m-%d"): v[0] for k, v in quarters.items()}
        self.assertEqual(values["2012-03-31"], 13.0)        # not the restated 99
        self.assertEqual(values["2012-12-31"], 60.0 - 39.0)  # not the tagged 17
        self.assertEqual(quarters[pd.Timestamp("2012-12-31")][1],
                         pd.Timestamp("2013-01-30"))         # 10-K filing date

    def test_q4_from_three_quarters_without_ytd(self):
        rows = [f for f in REVENUE_2012 if not (f.get("start") == "2012-01-01"
                                                and f["end"] == "2012-09-30")]
        rows = [f for f in rows if f["val"] != 17.0]
        doc = facts_doc({"SalesRevenueNet": rows})["facts"]["us-gaap"]
        q = eq.quarterly_flow(doc, "SalesRevenueNet")
        self.assertEqual(q[pd.Timestamp("2012-12-31")][0], 60.0 - (13 + 12 + 14))

    def test_revenue_concept_choice_and_point_in_time(self):
        doc = facts_doc({
            # Total and goods-only revenue filed together: take the total.
            "SalesRevenueNet": [fact("2013-01-01", "2013-03-31", 16.0, "2013-04-26")],
            "SalesRevenueGoodsNet": [fact("2013-01-01", "2013-03-31", 13.0, "2013-04-26")],
            # A later, larger restatement under a new concept: keep the first.
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                fact("2013-01-01", "2013-03-31", 18.0, "2014-04-25"),
                # A quarter first filed 500 days late: not point-in-time.
                fact("2013-04-01", "2013-06-30", 17.0, "2014-11-12"),
            ],
            "NetIncomeLoss": [fact("2013-01-01", "2013-03-31", 1.0, "2013-04-26")],
            "Assets": [fact(None, "2013-03-31", 100.0, "2013-04-26")],
            "LongTermDebtNoncurrent": [fact(None, "2013-03-31", 20.0, "2013-04-26")],
            "LongTermDebtCurrent": [fact(None, "2013-03-31", 5.0, "2013-04-26")],
            "CommercialPaper": [fact(None, "2013-03-31", 3.0, "2013-04-26")],
        })
        table = eq.company_quarters("X", doc).set_index("Date")
        q1 = table.loc[pd.Timestamp("2013-03-31").date()]
        self.assertEqual(q1["Total Revenue"], 16.0)
        self.assertEqual(q1["Revenue_Concept"], "SalesRevenueNet")
        self.assertTrue(q1["Point_In_Time"])
        self.assertEqual(q1["Total Debt"], 20.0 + 5.0 + 3.0)
        self.assertEqual(q1["Total Assets"], 100.0)
        q2 = table.loc[pd.Timestamp("2013-06-30").date()]
        self.assertFalse(q2["Point_In_Time"])


def quarterly_rows(revenues, start="2020-03-31", late=()):
    dates = pd.date_range(start, periods=len(revenues), freq="QE")
    return pd.DataFrame({
        "Date": dates.date,
        "Ticker": "X",
        "Filed": (dates + pd.Timedelta(days=35)).date,
        "Point_In_Time": [i not in late for i in range(len(revenues))],
        "Total Revenue": revenues,
        "Net Income": 1.0, "Operating Income": 2.0, "Total Assets": 50.0,
        "Total Debt": 10.0, "Stockholders Equity": 20.0,
        "Current Assets": 15.0, "Current Liabilities": 10.0,
    })


class LabelTests(unittest.TestCase):
    def test_qoq_and_yoy_labels(self):
        # Q1..Q6: 100, 90, 95, 120 | 98, 99
        labelled = label_company(quarterly_rows([100, 90, 95, 120, 98, 99]))
        qoq = labelled["Next_Period_Revenue_Decline"].tolist()
        self.assertEqual(qoq, [1.0, 0.0, 0.0, 1.0, 0.0])
        yoy = labelled["Next_Period_Revenue_Decline_YoY"].tolist()
        # Next quarter vs the same quarter a year earlier:
        # row Q4 -> next Q5 (98) vs Q1 (100): decline; row Q5 -> Q6 (99) vs Q2 (90): no.
        self.assertTrue(all(pd.isna(v) for v in yoy[:3]))
        self.assertEqual(yoy[3:], [1.0, 0.0])
        first = labelled.iloc[0]
        self.assertEqual(pd.Timestamp(first["Label_Available_Date"]),
                         pd.Timestamp(labelled.iloc[1]["Available_Date"]))

    def test_non_point_in_time_quarters_are_not_labelled(self):
        labelled = label_company(quarterly_rows([100, 90, 95, 120], late=(1,)))
        # Q1->Q2 and Q2->Q3 both involve the late Q2.
        self.assertEqual(len(labelled), 1)
        self.assertEqual(str(labelled.iloc[0]["Date"])[:10], "2020-09-30")

    def test_nearest_quarter_end(self):
        self.assertEqual(nearest_quarter_end("2024-01-28"), pd.Timestamp("2023-12-31"))
        self.assertEqual(nearest_quarter_end("2023-09-30"), pd.Timestamp("2023-09-30"))
        self.assertEqual(nearest_quarter_end("2024-06-29"), pd.Timestamp("2024-06-30"))


class SettingsTests(unittest.TestCase):
    def test_env_file_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# comment\nFOO_SETTING = 'from file'\n", encoding="utf-8")
            with mock.patch.object(settings, "ENV_FILE", env), \
                    mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FOO_SETTING", None)
                self.assertEqual(settings.get_setting("FOO_SETTING"), "from file")
                os.environ["FOO_SETTING"] = "from env"
                self.assertEqual(settings.get_setting("FOO_SETTING"), "from env")
                os.environ.pop("FOO_SETTING")
            self.assertEqual(settings.get_setting("MISSING_X", "d"), "d")

    def test_client_requires_contact_email(self):
        with self.assertRaises(EdgarConfigError):
            EdgarClient(user_agent="no contact here")


if __name__ == "__main__":
    unittest.main()
