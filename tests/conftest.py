"""Shared fixtures: a local stand-in for the UniProt REST API."""

from __future__ import annotations

import http.server
import random
import re
import socketserver
import threading

import pytest

AA = "ACDEFGHIKLMNPQRSTVWY"

# Real headers and real lengths for the three default targets; sequence content is
# synthesized because none of the code under test depends on it.
REAL_META = {
    "P21397": (
        "sp|P21397|AOFA_HUMAN Amine oxidase [flavin-containing] A OS=Homo sapiens "
        "OX=9606 GN=MAOA PE=1 SV=1",
        527,
    ),
    "P29274": (
        "sp|P29274|AA2AR_HUMAN Adenosine receptor A2a OS=Homo sapiens OX=9606 "
        "GN=ADORA2A PE=1 SV=2",
        412,
    ),
    "P21554": (
        "sp|P21554|CNR1_HUMAN Cannabinoid receptor 1 OS=Homo sapiens OX=9606 "
        "GN=CNR1 PE=1 SV=1",
        472,
    ),
}


def _fasta(header: str, seq: str) -> str:
    wrapped = "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))
    return f">{header}\n{wrapped}\n"


@pytest.fixture(scope="session")
def mock_uniprot():
    """
    Serve FASTA locally, including the failure modes the real API produces only rarely:
    404, a 200 with an empty body (obsolete/demerged accession), a 200 carrying an HTML
    error page, illegal residue characters, and a flaky endpoint that 500s twice.
    """
    rng = random.Random(7)
    bodies = {
        acc: _fasta(hdr, "".join(rng.choice(AA) for _ in range(n)))
        for acc, (hdr, n) in REAL_META.items()
    }
    state = {"flaky_hits": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence the test log
            pass

        def _send(self, payload: bytes, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            m = re.match(r"^/uniprotkb/([A-Za-z0-9_\-]+)\.fasta$", self.path)
            if not m:
                self.send_error(400)
                return
            acc = m.group(1)

            if acc in bodies:
                self._send(bodies[acc].encode())
            elif acc == "MISSING999":
                self.send_error(404)
            elif acc == "EMPTY200":
                self._send(b"")
            elif acc == "NOTFASTA":
                self._send(b"<html><body>Service unavailable</body></html>")
            elif acc == "BADALPHA":
                self._send(b">sp|TEST|BAD\nACDEFG123!!!ZZZ\n")
            elif acc == "FLAKY":
                state["flaky_hits"] += 1
                if state["flaky_hits"] <= 2:
                    self.send_error(500)
                else:
                    self._send(bodies["P29274"].encode())
            else:
                self.send_error(404)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = Server(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/uniprotkb/{{}}.fasta"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def patched_uniprot(mock_uniprot, monkeypatch):
    """Point dti_screen.targets at the mock server for the duration of a test."""
    from dti_screen import targets

    monkeypatch.setattr(targets, "UNIPROT_FASTA_URL", mock_uniprot)
    return mock_uniprot
