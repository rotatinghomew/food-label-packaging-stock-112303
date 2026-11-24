from pathlib import Path
import unittest

from relabel.loader import load_dataset
from relabel.ledger import Ledger, shipment_state

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AS_OF = "2026-09-05T08:00"


class LedgerRangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(DATA)
        cls.ledger = Ledger(cls.data, AS_OF)

    def test_count_collapses_prior_uncertainty(self):
        a = self.ledger.old_range("A", "dr1", AS_OF)
        self.assertEqual((a.total, a.old_min, a.old_max), (60, 43, 43))

        b = self.ledger.old_range("B", "dr1", AS_OF)
        self.assertEqual((b.total, b.old_min, b.old_max), (92, 19, 33))
        self.assertEqual(b.confirmed_old, 19)
        self.assertEqual(b.possible_extra_old, 14)

        c = self.ledger.old_range("C", "dr1", AS_OF)
        self.assertEqual((c.total, c.old_min, c.old_max), (125, 32, 91))
        self.assertEqual(c.possible_extra_old, 59)

    def test_transfer_old_fraction_is_shared_across_stores(self):
        self.assertEqual(self.ledger.shipment_old_range("x1"), (10, 36))
        self.assertEqual(self.ledger.shipment_old_range("x2"), (0, 14))
        low, high = self.ledger.global_old_range("dr1", AS_OF)
        self.assertEqual((low, high), (94, 167))
        network_low, network_high = self.ledger.global_old_range(
            "dr1", AS_OF, include_in_transit=True
        )
        self.assertEqual((network_low, network_high), (108, 167))

    def test_in_transit_quantity_is_not_sellable_at_either_end(self):
        # x1 picked at 9/2 07, before arrival on 9/3 11.
        self.assertEqual(self.ledger.total_qty("A", "dr1", __import__("datetime").datetime.fromisoformat("2026-09-02T12:00")), 60)
        self.assertEqual(self.ledger.total_qty("B", "dr1", "2026-09-02T12:00"), 90)
        # After arrival B includes it; A has already excluded it.
        self.assertEqual(self.ledger.total_qty("B", "dr1", "2026-09-03T12:00"), 126)

    def test_pending_return_is_not_sellable_and_not_used_as_sale_reconciliation(self):
        stock = self.ledger.old_range("B", "dr1", "2026-09-03T12:30")
        self.assertEqual(stock.total, 126)
        self.assertEqual(stock.inspection_units, 8)
        counted = self.ledger.old_range("B", "dr1", "2026-09-03T15:00")
        self.assertEqual((counted.total, counted.old_min, counted.old_max), (106, 33, 33))

    def test_late_known_sale_is_excluded_from_early_decision_snapshot(self):
        early = Ledger(self.data, "2026-09-06T12:00")
        a = early.old_range("A", "dr1", "2026-09-06T12:00")
        self.assertEqual((a.total, a.old_min, a.old_max), (60, 43, 43))

        late = Ledger(self.data, "2026-09-08T20:00")
        before_removal = late.old_range("A", "dr1", "2026-09-06T16:59")
        self.assertEqual((before_removal.total, before_removal.old_min, before_removal.old_max), (55, 38, 43))
        after_removal = late.old_range("A", "dr1", "2026-09-06T18:00")
        self.assertEqual((after_removal.total, after_removal.old_min, after_removal.old_max), (27, 10, 15))

    def test_unconfirmed_transfer_state(self):
        x2 = next(r for r in self.data.transfers if r["shipment"] == "x2")
        self.assertEqual(shipment_state(x2, AS_OF, AS_OF), "在途")
        x1 = next(r for r in self.data.transfers if r["shipment"] == "x1")
        self.assertEqual(shipment_state(x1, "2026-09-03T12:00", "2026-09-03T12:00"), "已送达待交接确认")
        self.assertEqual(shipment_state(x1, AS_OF, AS_OF), "已交接确认")


if __name__ == "__main__":
    unittest.main()
