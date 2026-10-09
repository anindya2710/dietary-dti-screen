"""
UniProt acquisition tests.

The important cases are the ones where the API returns HTTP 200 with something that is
not a usable sequence. A naive fetcher passes those downstream and silently screens
against an empty or garbage target.
"""

from __future__ import annotations

import pytest

from dti_screen.targets import fetch_target, fetch_targets, parse_fasta


class TestParseFasta:
    def test_parses_single_record(self):
        header, seq = parse_fasta(">sp|P1|TEST desc\nACDEF\nGHIKL\n")
        assert header == "sp|P1|TEST desc"
        assert seq == "ACDEFGHIKL"

    def test_strips_whitespace_and_uppercases(self):
        _, seq = parse_fasta(">h\n  acd ef \n ghi \n")
        assert seq == "ACDEFGHI"

    @pytest.mark.parametrize(
        "body, match",
        [
            ("", "empty response body"),
            ("   \n  \n", "empty response body"),
            ("ACDEFG\n", "not FASTA"),
            ("<html>nope</html>", "not FASTA"),
            (">header only\n", "sequence is empty"),
        ],
    )
    def test_rejects_bad_bodies(self, body, match):
        with pytest.raises(ValueError, match=match):
            parse_fasta(body)


class TestFetchTarget:
    def test_valid_accession(self, patched_uniprot, tmp_path):
        rec = fetch_target("P21397", label="MAOA", cache_dir=str(tmp_path))
        assert rec.label == "MAOA"
        assert rec.uniprot_id == "P21397"
        assert rec.length == 527  # real length of MAOA
        assert "MAOA" in rec.header
        assert rec.cached is False

    def test_404_raises_runtime_error(self, patched_uniprot, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            fetch_target("MISSING999", cache_dir=str(tmp_path))

    def test_empty_200_is_rejected(self, patched_uniprot, tmp_path):
        """An obsolete accession can return 200 with no body."""
        with pytest.raises(ValueError, match="empty response body"):
            fetch_target("EMPTY200", cache_dir=str(tmp_path))

    def test_html_error_page_with_200_is_rejected(self, patched_uniprot, tmp_path):
        with pytest.raises(ValueError, match="not FASTA"):
            fetch_target("NOTFASTA", cache_dir=str(tmp_path))

    def test_illegal_residues_are_rejected(self, patched_uniprot, tmp_path):
        with pytest.raises(ValueError, match="non-amino-acid"):
            fetch_target("BADALPHA", cache_dir=str(tmp_path))

    def test_retries_transient_500s(self, patched_uniprot, tmp_path):
        """Two 500s then a 200 must succeed via the adapter's retry policy."""
        rec = fetch_target("FLAKY", cache_dir=str(tmp_path))
        assert rec.length == 412

    def test_second_fetch_is_served_from_cache(self, patched_uniprot, tmp_path):
        first = fetch_target("P21554", cache_dir=str(tmp_path))
        second = fetch_target("P21554", cache_dir=str(tmp_path))
        assert first.cached is False
        assert second.cached is True
        assert first.sequence == second.sequence

    def test_works_without_a_cache_dir(self, patched_uniprot):
        rec = fetch_target("P29274", cache_dir=None)
        assert rec.length == 412

    def test_nonstandard_residue_count(self, patched_uniprot, tmp_path):
        rec = fetch_target("P21397", cache_dir=str(tmp_path))
        assert rec.n_nonstandard == 0


class TestFetchTargets:
    def test_one_bad_accession_does_not_block_the_others(self, patched_uniprot, tmp_path):
        records, failures = fetch_targets(
            {"MAOA": "P21397", "BROKEN": "MISSING999", "CB1": "P21554"},
            cache_dir=str(tmp_path),
            verbose=False,
        )
        assert set(records) == {"MAOA", "CB1"}
        assert set(failures) == {"BROKEN"}
        assert "does not exist" in failures["BROKEN"]

    def test_all_three_default_targets(self, patched_uniprot, tmp_path):
        from dti_screen.config import DEFAULT_TARGETS

        records, failures = fetch_targets(
            DEFAULT_TARGETS, cache_dir=str(tmp_path), verbose=False
        )
        assert not failures
        assert {r.length for r in records.values()} == {527, 412, 472}

    def test_default_targets_are_within_cnn_limit(self, patched_uniprot, tmp_path):
        """All three must avoid CNN protein-encoder truncation."""
        from dti_screen.chem import MAX_SEQ_PROTEIN
        from dti_screen.config import DEFAULT_TARGETS

        records, _ = fetch_targets(DEFAULT_TARGETS, cache_dir=str(tmp_path), verbose=False)
        for rec in records.values():
            assert rec.length <= MAX_SEQ_PROTEIN
