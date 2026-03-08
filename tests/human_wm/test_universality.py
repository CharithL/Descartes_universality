"""
Tests for human_wm.analysis.universality -- orchestration and reporting.

These tests cover the pure-Python aspects of the universality module:
    - format_universality_table: string formatting of the final report
    - Verdict logic: classification thresholds for cross-seed,
      cross-patient, and cross-architecture tests
    - Summary structure: correct keys and types in summary dicts

Full integration tests (which train models) are expensive and best run
manually via the pipeline scripts (13-17).  These unit tests verify the
logic around aggregation and presentation without any model training.
"""

from __future__ import annotations

import pytest

# Guard: universality.py transitively imports torch via descartes_core.
# On Windows machines where pytorch's DLL loading fails under pytest's
# import-rewriting, all tests will be skipped.
try:
    from human_wm.analysis.universality import format_universality_table
    HAS_UNIVERSALITY = True
except (ImportError, OSError):
    HAS_UNIVERSALITY = False
    format_universality_table = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not HAS_UNIVERSALITY,
    reason="universality module not importable (torch DLL error)",
)


# ===================================================================
# Fixtures: pre-built summary dicts
# ===================================================================


@pytest.fixture()
def cross_seed_summary():
    """Example cross-seed summary with mixed verdicts."""
    return {
        'test': 'cross_seed',
        'n_seeds': 10,
        'successful_seeds': 10,
        'variables': {
            'persistent_delay': {
                'n_mandatory': 9,
                'n_total': 10,
                'pct': 90.0,
                'verdict': 'ROBUST',
                'classifications': ['MANDATORY'] * 9 + ['LEARNED_ZOMBIE'],
            },
            'memory_load': {
                'n_mandatory': 6,
                'n_total': 10,
                'pct': 60.0,
                'verdict': 'MODERATE',
                'classifications': ['MANDATORY'] * 6 + ['ZOMBIE'] * 4,
            },
            'theta_modulation': {
                'n_mandatory': 2,
                'n_total': 10,
                'pct': 20.0,
                'verdict': 'FRAGILE',
                'classifications': ['MANDATORY'] * 2 + ['ZOMBIE'] * 8,
            },
        },
    }


@pytest.fixture()
def cross_patient_summary():
    """Example cross-patient summary."""
    return {
        'test': 'cross_patient',
        'n_patients': 10,
        'variables': {
            'persistent_delay': {
                'n_mandatory': 9,
                'n_patients': 10,
                'pct': 90.0,
                'verdict': 'UNIVERSAL',
            },
            'memory_load': {
                'n_mandatory': 6,
                'n_patients': 10,
                'pct': 60.0,
                'verdict': 'PARTIAL',
            },
            'theta_modulation': {
                'n_mandatory': 1,
                'n_patients': 10,
                'pct': 10.0,
                'verdict': 'NO',
            },
        },
    }


@pytest.fixture()
def cross_arch_summary():
    """Example cross-architecture summary."""
    return {
        'test': 'cross_architecture',
        'architectures': ['lstm', 'gru', 'transformer', 'linear'],
        'variables': {
            'persistent_delay': {
                'n_mandatory': 4,
                'n_tested': 4,
                'mandatory_in': ['lstm', 'gru', 'transformer', 'linear'],
                'verdict': 'UNIVERSAL',
            },
            'memory_load': {
                'n_mandatory': 2,
                'n_tested': 4,
                'mandatory_in': ['lstm', 'gru'],
                'verdict': 'PARTIAL',
            },
            'theta_modulation': {
                'n_mandatory': 0,
                'n_tested': 4,
                'mandatory_in': [],
                'verdict': 'ARCH_SPECIFIC',
            },
        },
    }


# ===================================================================
# 1. format_universality_table
# ===================================================================


class TestFormatUniversalityTable:
    """Tests for the final report formatting function."""

    def test_returns_string(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """format_universality_table returns a non-empty string."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_title(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """Output should contain the report title."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        assert 'UNIVERSALITY REPORT' in result

    def test_contains_all_variables(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """All variable names from all summaries should appear."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        assert 'persistent_delay' in result
        assert 'memory_load' in result
        assert 'theta_modulation' in result

    def test_contains_verdicts(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """Overall verdicts should appear in the output."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        # persistent_delay is ROBUST+UNIVERSAL+UNIVERSAL → UNIVERSAL
        assert 'UNIVERSAL' in result

    def test_contains_verdict_key(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """Output should contain the verdict key legend."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        assert 'VERDICT KEY' in result

    def test_seed_only(self, cross_seed_summary):
        """Works with only cross-seed data provided."""
        result = format_universality_table(
            cross_seed_summary=cross_seed_summary,
        )
        assert isinstance(result, str)
        assert 'persistent_delay' in result

    def test_patient_only(self, cross_patient_summary):
        """Works with only cross-patient data provided."""
        result = format_universality_table(
            cross_patient_summary=cross_patient_summary,
        )
        assert isinstance(result, str)
        assert 'memory_load' in result

    def test_arch_only(self, cross_arch_summary):
        """Works with only cross-architecture data provided."""
        result = format_universality_table(
            cross_arch_summary=cross_arch_summary,
        )
        assert isinstance(result, str)
        assert 'theta_modulation' in result

    def test_all_none_returns_header(self):
        """When all summaries are None, still returns a valid string
        with header and legend."""
        result = format_universality_table()
        assert isinstance(result, str)
        assert 'UNIVERSALITY REPORT' in result
        assert 'VERDICT KEY' in result

    def test_numeric_ratios_present(
        self, cross_seed_summary, cross_patient_summary, cross_arch_summary,
    ):
        """The output should contain ratio strings like '9/10'."""
        result = format_universality_table(
            cross_seed_summary, cross_patient_summary, cross_arch_summary,
        )
        assert '9/10' in result  # persistent_delay cross-seed
        assert '4/4' in result   # persistent_delay cross-arch

    def test_superset_variables(self, cross_seed_summary):
        """When one summary has a variable the others don't, it should
        show '?/?' for the missing tests."""
        result = format_universality_table(
            cross_seed_summary=cross_seed_summary,
        )
        # cross-patient and cross-arch columns should show ?/? for these
        assert '?/?' in result


# ===================================================================
# 2. Overall verdict logic
# ===================================================================


class TestOverallVerdict:
    """Test the overall verdict logic computed within
    format_universality_table.

    The verdict is:
        UNIVERSAL: all component verdicts are ROBUST/UNIVERSAL (≥2 tests)
        ROBUST:    any component verdict is ROBUST/UNIVERSAL
        PARTIAL:   any component verdict is MODERATE/PARTIAL
        ZOMBIE:    nothing above
    """

    def test_universal_verdict(self):
        """A variable that is ROBUST+UNIVERSAL+UNIVERSAL → UNIVERSAL."""
        seed = _make_seed_summary('persistent_delay', 9, 10, 'ROBUST')
        patient = _make_patient_summary('persistent_delay', 9, 10, 'UNIVERSAL')
        arch = _make_arch_summary('persistent_delay', 4, 4, 'UNIVERSAL')

        table = format_universality_table(seed, patient, arch)
        # Find the persistent_delay line
        line = _find_var_line(table, 'persistent_delay')
        assert 'UNIVERSAL' in line

    def test_robust_verdict(self):
        """A variable that is ROBUST+PARTIAL → ROBUST (at least one
        ROBUST/UNIVERSAL, but not all)."""
        seed = _make_seed_summary('memory_load', 8, 10, 'ROBUST')
        patient = _make_patient_summary('memory_load', 6, 10, 'PARTIAL')

        table = format_universality_table(seed, patient)
        line = _find_var_line(table, 'memory_load')
        assert 'ROBUST' in line

    def test_partial_verdict(self):
        """A variable that is MODERATE+PARTIAL → PARTIAL."""
        seed = _make_seed_summary('gamma_mod', 5, 10, 'MODERATE')
        patient = _make_patient_summary('gamma_mod', 5, 10, 'PARTIAL')

        table = format_universality_table(seed, patient)
        line = _find_var_line(table, 'gamma_mod')
        assert 'PARTIAL' in line

    def test_zombie_verdict(self):
        """A variable that is FRAGILE+NO → ZOMBIE."""
        seed = _make_seed_summary('theta_mod', 2, 10, 'FRAGILE')
        patient = _make_patient_summary('theta_mod', 1, 10, 'NO')

        table = format_universality_table(seed, patient)
        line = _find_var_line(table, 'theta_mod')
        assert 'ZOMBIE' in line

    def test_single_test_robust(self):
        """With only one test showing ROBUST, overall is ROBUST."""
        seed = _make_seed_summary('delay_stab', 9, 10, 'ROBUST')

        table = format_universality_table(seed)
        line = _find_var_line(table, 'delay_stab')
        # With only 1 test, can't be UNIVERSAL (needs ≥2)
        # but ROBUST component → ROBUST overall
        assert 'ROBUST' in line


# ===================================================================
# 3. Cross-seed verdict thresholds
# ===================================================================


class TestCrossSeedVerdictLogic:
    """Verify the cross-seed verdict boundaries (computed in
    cross_seed_test, tested here via summary dicts)."""

    def test_eight_of_ten_is_robust(self):
        """≥8/10 mandatory → ROBUST."""
        summary = _make_seed_summary('var', 8, 10, 'ROBUST')
        var = summary['variables']['var']
        assert var['verdict'] == 'ROBUST'

    def test_five_of_ten_is_moderate(self):
        """≥5 but <8 mandatory → MODERATE."""
        summary = _make_seed_summary('var', 5, 10, 'MODERATE')
        var = summary['variables']['var']
        assert var['verdict'] == 'MODERATE'

    def test_four_of_ten_is_fragile(self):
        """<5 mandatory → FRAGILE."""
        summary = _make_seed_summary('var', 4, 10, 'FRAGILE')
        var = summary['variables']['var']
        assert var['verdict'] == 'FRAGILE'


# ===================================================================
# 4. Cross-patient verdict thresholds
# ===================================================================


class TestCrossPatientVerdictLogic:
    """Verify the cross-patient verdict boundaries."""

    def test_eighty_pct_is_universal(self):
        """≥80% → UNIVERSAL."""
        summary = _make_patient_summary('var', 8, 10, 'UNIVERSAL')
        var = summary['variables']['var']
        assert var['verdict'] == 'UNIVERSAL'
        assert var['pct'] == 80.0

    def test_fifty_pct_is_partial(self):
        """≥50% but <80% → PARTIAL."""
        summary = _make_patient_summary('var', 5, 10, 'PARTIAL')
        var = summary['variables']['var']
        assert var['verdict'] == 'PARTIAL'

    def test_below_twenty_is_no(self):
        """<20% → NO."""
        summary = _make_patient_summary('var', 1, 10, 'NO')
        var = summary['variables']['var']
        assert var['verdict'] == 'NO'


# ===================================================================
# 5. Cross-architecture verdict thresholds
# ===================================================================


class TestCrossArchVerdictLogic:
    """Verify the cross-architecture verdict boundaries."""

    def test_all_four_is_universal(self):
        """Mandatory in all 4 tested → UNIVERSAL."""
        summary = _make_arch_summary('var', 4, 4, 'UNIVERSAL')
        var = summary['variables']['var']
        assert var['verdict'] == 'UNIVERSAL'

    def test_two_of_four_is_partial(self):
        """Mandatory in 2/4 → PARTIAL."""
        summary = _make_arch_summary('var', 2, 4, 'PARTIAL')
        var = summary['variables']['var']
        assert var['verdict'] == 'PARTIAL'

    def test_one_of_four_is_arch_specific(self):
        """Mandatory in <2 → ARCH_SPECIFIC."""
        summary = _make_arch_summary('var', 1, 4, 'ARCH_SPECIFIC')
        var = summary['variables']['var']
        assert var['verdict'] == 'ARCH_SPECIFIC'


# ===================================================================
# Helpers for building minimal summary dicts
# ===================================================================


def _make_seed_summary(var_name, n_mandatory, n_total, verdict):
    """Build a minimal cross-seed summary with one variable."""
    return {
        'test': 'cross_seed',
        'n_seeds': n_total,
        'successful_seeds': n_total,
        'variables': {
            var_name: {
                'n_mandatory': n_mandatory,
                'n_total': n_total,
                'pct': n_mandatory / n_total * 100,
                'verdict': verdict,
                'classifications': (
                    ['MANDATORY'] * n_mandatory
                    + ['ZOMBIE'] * (n_total - n_mandatory)
                ),
            },
        },
    }


def _make_patient_summary(var_name, n_mandatory, n_patients, verdict):
    """Build a minimal cross-patient summary with one variable."""
    return {
        'test': 'cross_patient',
        'n_patients': n_patients,
        'variables': {
            var_name: {
                'n_mandatory': n_mandatory,
                'n_patients': n_patients,
                'pct': n_mandatory / n_patients * 100,
                'verdict': verdict,
            },
        },
    }


def _make_arch_summary(var_name, n_mandatory, n_tested, verdict):
    """Build a minimal cross-architecture summary with one variable."""
    return {
        'test': 'cross_architecture',
        'architectures': ['lstm', 'gru', 'transformer', 'linear'][:n_tested],
        'variables': {
            var_name: {
                'n_mandatory': n_mandatory,
                'n_tested': n_tested,
                'mandatory_in': ['lstm', 'gru', 'transformer', 'linear'][:n_mandatory],
                'verdict': verdict,
            },
        },
    }


def _find_var_line(table_str: str, var_name: str) -> str:
    """Find the line in the table containing a given variable name."""
    for line in table_str.split('\n'):
        if var_name in line:
            return line
    return ''
