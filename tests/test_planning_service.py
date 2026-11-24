import unittest
from pathlib import Path

from relabel.forecast import DemandModel, residual_envelope
from relabel.loader import load_dataset
from relabel.service import ScenarioService

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AS_OF = "2026-09-05T08:00"


class ForecastTest(unittest.TestCase):
    def test_short_day_uses_available_exposure_not_zero_tail(self):
        model = DemandModel(load_dataset(DATA))
        # 18:00-08:00 = 10h; 14:00-08:00 = 6h.
        self.assertAlmostEqual(model.rate("A"), 3.743055555555556, places=6)
        forecast = model.forecast("A", AS_OF, "2026-09-06T18:00")
        self.assertEqual(forecast.open_hours, 20.0)
        self.assertGreater(forecast.mean_demand, 60)

    def test_backtest_has_all_16_trials_and_three_stores(self):
        model = DemandModel(load_dataset(DATA))
        result = model.backtest_trials()
        self.assertEqual(result["leave_one_trial_out"]["n"], 48)
        self.assertGreater(result["leave_one_trial_out"]["mae"], 0)
        self.assertIn("A", model.old_end_calibration())

    def test_residual_envelope_separates_sources(self):
        env = residual_envelope(
            total=10,
            old_min=4,
            old_max=6,
            demand_low=1,
            demand_median=5,
            demand_high=8,
        )
        self.assertEqual((env["combined_min"], env["combined_max"]), (0, 6))
        self.assertEqual(env["stock_uncertainty_at_median_demand"], [0, 5])
        self.assertEqual(env["current_stock_width"], 2)


class PlanningAndServiceTest(unittest.TestCase):
    def setUp(self):
        self.data = load_dataset(DATA)
        self.service = ScenarioService(self.data)
        self.report = self.service.planning_report(AS_OF)

    def test_supplied_ranges_are_reported(self):
        by_store = {p["store"]: p for p in self.report["stores"]}
        self.assertEqual(by_store["A"]["stock"]["old_max"], 43)
        self.assertEqual((by_store["B"]["stock"]["old_min"], by_store["B"]["stock"]["old_max"]), (19, 33))
        self.assertEqual((by_store["C"]["stock"]["old_min"], by_store["C"]["stock"]["old_max"]), (32, 91))

    def test_no_store_can_use_forecast_as_zero_old_exemption(self):
        for plan in self.report["stores"]:
            self.assertFalse(plan["exemption"]["can_certify_zero"])

    def test_unconfirmed_outgoing_transfer_blocks_zero_cancellation_guard(self):
        b = next(p for p in self.report["stores"] if p["store"] == "B")
        self.assertTrue(b["appointment_guard"]["cannot_cancel_as_zero"])
        self.assertEqual([x["shipment"] for x in b["appointment_guard"]["unconfirmed_outgoing"]], ["x2"])

    def test_count_ranking_and_crew_overlap(self):
        slots = {c["slot"]: c for c in self.report["count_slots"]}
        self.assertEqual(slots["s3"]["stock_ambiguity_units"], 59)
        self.assertEqual(slots["s2"]["resolves_shipment_provenance"], ["x2"])
        self.assertEqual(slots["s1"]["resolves_shipment_provenance"], [])
        self.assertEqual(self.report["parallel_count_plan"]["selected_slots"], ["s1", "s2", "s3"])
        self.assertEqual(self.report["single_crew_count_plan"]["selected_slots"], ["s1", "s2"])

    def test_vehicle_total_capacity_is_not_overbooked(self):
        for vehicle in self.report["vehicles"]["vehicles"]:
            allocated = sum(a["allocated"] for a in vehicle["allocations"])
            self.assertLessEqual(allocated, vehicle["capacity"])
            self.assertEqual(allocated + vehicle["remaining_capacity"], vehicle["capacity"])
            by_store = {a["store"]: a for a in vehicle["allocations"]}
            self.assertTrue(by_store["A"]["latest_confirmation_before_vehicle_open"])
            self.assertFalse(by_store["C"]["latest_confirmation_before_vehicle_open"])

    def test_actual_zero_count_after_sale_can_certify_exemption(self):
        # n1 已证明 A 在 9/2 10 有 43 旧/17 新；再补一笔 43 件全旧收银并实盘为 0 旧/17 新。
        self.service.add_movement(
            "test-sale", "A", "dr1",
            "2026-09-05T10:30", "2026-09-05T10:30",
            "收银", 43, 43, 43,
        )
        self.service.add_count(
            "test-count", "A",
            "2026-09-05T11:00", "2026-09-05T11:00",
            0, 17,
        )
        report = self.service.planning_report("2026-09-05T12:00")
        a = next(p for p in report["stores"] if p["store"] == "A")
        self.assertTrue(a["exemption"]["can_certify_zero"])
        self.assertEqual(a["exemption"]["basis_count"], "test-count")

    def test_handoff_confirmation_changes_state_without_adding_sellable_before_arrival(self):
        x2 = next(r for r in self.data.transfers if r["shipment"] == "x2")
        self.assertIsNone(x2["known"])
        self.service.record_handoff_confirmation("x2", "2026-09-05T09:00")
        # 没有 arrived 时间，确认单不能把在途货变成 C 可售。
        c_total = self.service.ledger("2026-09-05T09:30").total_qty("C", "dr1", "2026-09-05T09:30")
        self.assertEqual(c_total, 125)
        self.service.record_arrival("x2", "2026-09-05T10:00")
        c_total = self.service.ledger("2026-09-05T10:30").total_qty("C", "dr1", "2026-09-05T10:30")
        self.assertEqual(c_total, 139)

    def test_removal_confirmation_deducts_exact_old_units(self):
        self.service.confirm_removal("test-rm", "B", "dr1", "2026-09-06T16:00", 19)
        stock = self.service.stock("B", "dr1", "2026-09-06T17:00")
        self.assertEqual(stock.total, 73)


if __name__ == "__main__":
    unittest.main()
