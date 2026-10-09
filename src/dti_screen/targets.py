"""
Protein target acquisition from the UniProt REST API.

The only subtle part is validation. UniProt can return HTTP 200 with a body that is
not a usable sequence -- an empty payload for an obsolete or demerged accession, or an
HTML error page from an upstream proxy. A bare ``requests.get(url).text`` passes both
of those downstream, where they become a silently wrong screen rather than an error.
Every response is therefore parsed and checked before it is accepted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 v2 / v1 compatibility
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

#: Overridable so tests can point at a local server.
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"

#: The 20 standard proteinogenic residues.
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: Ambiguity codes UniProt may legitimately emit; allowed, but counted and reported.
AA_TOLERATED = AA_ALPHABET | frozenset("BXZUO")


@dataclass(frozen=True)
class TargetRecord:
    """A validated protein target."""

    label: str
    uniprot_id: str
    header: str
    sequence: str
    cached: bool = False

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_nonstandard(self) -> int:
        return sum(1 for c in self.sequence if c not in AA_ALPHABET)


def make_session(max_retries: int = 4, user_agent: str = "dti-screen/1.0") -> requests.Session:
    """A ``requests.Session`` that retries transient failures with exponential backoff."""
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=1.0,  # 0s, 1s, 2s, 4s ...
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    # Renamed in urllib3 1.26; set whichever attribute exists.
    try:
        retry.allowed_methods = frozenset(["GET"])
    except AttributeError:  # pragma: no cover
        retry.method_whitelist = frozenset(["GET"])  # type: ignore[attr-defined]

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def parse_fasta(text: str) -> tuple[str, str]:
    """
    Parse a single-record FASTA document into ``(header, sequence)``.

    Raises ``ValueError`` on an empty body, a body that is not FASTA, or a record whose
    header is present but whose sequence is empty.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty response body")
    if not lines[0].startswith(">"):
        raise ValueError(f"not FASTA; body starts with {lines[0][:60]!r}")
    header = lines[0].lstrip(">")
    sequence = "".join(lines[1:]).upper().replace(" ", "")
    if not sequence:
        raise ValueError("FASTA header present but sequence is empty")
    return header, sequence


def fetch_target(
    uniprot_id: str,
    label: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
    cache_dir: str | None = None,
) -> TargetRecord:
    """
    Fetch and validate the canonical sequence for one UniProt accession.

    Responses are cached on disk by accession so repeated runs do not re-hit the API.
    Raises ``RuntimeError`` for HTTP-level failures and ``ValueError`` for bodies that
    are not a usable protein sequence.
    """
    label = label or uniprot_id
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{uniprot_id}.fasta")

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            header, sequence = parse_fasta(fh.read())
        cached = True
    else:
        session = session or make_session()
        url = UNIPROT_FASTA_URL.format(uniprot_id)
        resp = session.get(url, timeout=timeout)

        if resp.status_code == 404:
            raise RuntimeError(f"{uniprot_id}: 404 - accession does not exist")
        if resp.status_code != 200:
            raise RuntimeError(f"{uniprot_id}: HTTP {resp.status_code}")

        header, sequence = parse_fasta(resp.text)  # rejects empty / non-FASTA 200s
        if cache_path:
            with open(cache_path, "w") as fh:
                fh.write(resp.text)
        cached = False

    illegal = sorted(set(sequence) - AA_TOLERATED)
    if illegal:
        raise ValueError(f"{uniprot_id}: non-amino-acid characters in sequence: {illegal}")

    return TargetRecord(
        label=label,
        uniprot_id=uniprot_id,
        header=header,
        sequence=sequence,
        cached=cached,
    )


def fetch_targets(
    targets: dict[str, str],
    cache_dir: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, TargetRecord], dict[str, str]]:
    """
    Fetch every target in ``{label: accession}``.

    One failing accession must not prevent the others from being screened, so failures
    are collected and returned rather than raised.
    """
    from .chem import MAX_SEQ_PROTEIN  # local import avoids a cycle at module load

    session = make_session()
    records: dict[str, TargetRecord] = {}
    failures: dict[str, str] = {}

    for label, accession in targets.items():
        try:
            rec = fetch_target(accession, label=label, session=session, cache_dir=cache_dir)
            records[label] = rec
            if verbose:
                notes = []
                if rec.length > MAX_SEQ_PROTEIN:
                    notes.append(f"WILL BE TRUNCATED to {MAX_SEQ_PROTEIN} by the CNN encoder")
                if rec.n_nonstandard:
                    notes.append(f"{rec.n_nonstandard} non-standard residues")
                if rec.cached:
                    notes.append("cached")
                suffix = ("  [" + "; ".join(notes) + "]") if notes else ""
                print(f"[ok]   {label:10s} {accession}  {rec.length:4d} aa{suffix}")
                print(f"       {rec.header[:95]}")
        except Exception as exc:  # noqa: BLE001 - per-target isolation is the point
            failures[label] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[FAIL] {label:10s} {accession}  {type(exc).__name__}: {exc}")

    return records, failures
