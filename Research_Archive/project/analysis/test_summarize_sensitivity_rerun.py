"""Arithmetic-only tests; synthetic rows are never written as research results."""

import importlib.util
import itertools
import math
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sensitivity_summary", Path(__file__).with_name("summarize_sensitivity_rerun.py")
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture():
    rows = []
    for d, seed, method, budget, qcut, rcut in itertools.product(
        summary.DIMENSIONS,
        summary.SEEDS,
        summary.METHODS,
        summary.BUDGETS,
        summary.QCUTS,
        summary.RCUTS,
    ):
        row = dict(
            dimension=d,
            seed=seed,
            method=method,
            refinement_steps=budget,
            quality_cutoff=qcut,
            region_cutoff=rcut,
            n_sets=30,
            n_candidates=8,
            gradient_queries_per_set=budget * 8,
            initial_matched_cosine=0.0,
        )
        row.update({metric: 0.0 for metric in summary.METRICS})
        row["candidate_background_rate"] = 1.0
        rows.append(row)
    caps = []
    for seed, method, condition in itertools.product(
        summary.SEEDS, summary.CAP_METHODS, ("uncapped", "cap6")
    ):
        row = dict(
            seed=seed,
            method=method,
            cap_condition=condition,
            n_sets=30,
            n_candidates=8,
            uncapped_max_displacement=7.0,
            uncapped_fraction_above6=1.0,
            used_max_displacement=7.0 if condition == "uncapped" else 6.0,
        )
        for stage in ("before", "after"):
            row.update({stage + "_" + metric: 0.0 for metric in summary.METRICS})
            row[stage + "_candidate_background_rate"] = 1.0
        caps.append(row)
    return rows, caps


class SummaryTests(unittest.TestCase):
    def test_sample_sd_not_population_sd(self):
        result = summary.stats(range(30))
        self.assertAlmostEqual(result["mean"], 14.5)
        self.assertAlmostEqual(result["sd_sample"], math.sqrt(77.5))

    def test_wrong_seed_count_rejected(self):
        with self.assertRaises(ValueError):
            summary.stats(range(29))

    def test_full_grid_exact_identity(self):
        rows, caps = fixture()
        _, checks = summary.validate_grid(rows, caps)
        self.assertEqual(checks["sensitivity_rows"], 8640)
        self.assertEqual(checks["cap_rows"], 180)

    def test_duplicate_grid_rejected(self):
        rows, caps = fixture()
        rows[-1] = dict(rows[0])
        with self.assertRaises(ValueError):
            summary.validate_grid(rows, caps)

    def test_cap_violation_rejected(self):
        rows, caps = fixture()
        next(r for r in caps if r["cap_condition"] == "cap6")[
            "used_max_displacement"
        ] = 6.1
        with self.assertRaises(ValueError):
            summary.validate_grid(rows, caps)

    def test_outcome_partition_rejected(self):
        rows, caps = fixture()
        rows[0]["candidate_useful_rate"] = 0.1
        with self.assertRaises(ValueError):
            summary.validate_grid(rows, caps)

    def test_cap_changes_are_paired_with_correct_sign(self):
        rows, caps = fixture()
        lookup, _ = summary.validate_grid(rows, caps)
        outputs = summary.summaries(lookup, caps)
        changes = outputs[4]
        selected = [r for r in changes if r["metric"] == "used_max_displacement"]
        self.assertEqual(len(selected), 3)
        self.assertTrue(
            all(r["mean"] == -1.0 and r["sd_sample"] == 0.0 for r in selected)
        )

    def test_table_values_are_computed_not_paper_substitutes(self):
        rows, caps = fixture()
        lookup, _ = summary.validate_grid(rows, caps)
        refinements = summary.summaries(lookup, caps)[0]
        table, comparisons, _ = summary.table_b2(refinements)
        self.assertTrue(all(row["coverage_9_mean"] == 0.0 for row in table))
        self.assertTrue(any(not row["same_at_four_decimals"] for row in comparisons))


if __name__ == "__main__":
    unittest.main(verbosity=2)
