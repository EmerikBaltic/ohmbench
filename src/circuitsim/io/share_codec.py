"""Qt-free URL-safe circuit sharing codec (`ohmbench://c/<zlib+base64>`)."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
import zlib

from circuitsim.io.circ_serializer import CircuitDocument

if TYPE_CHECKING:
    from circuitsim.wiring.serializer import CircuitDocumentV2

SHARE_URL_SCHEME: str = "ohmbench"
SHARE_URL_PREFIX: str = "ohmbench://c/"
_MAX_DECOMPRESSED_BYTES: int = 16 * 1024 * 1024


class ShareUrlError(ValueError):
    """Raised when an `ohmbench://c/...` share URL cannot be decoded or validated."""


def encode_circuit_share_url(doc: CircuitDocumentV2 | CircuitDocument) -> str:
    """Encode a schematic document into a compact `ohmbench://c/...` URL string."""
    from circuitsim.wiring.serializer import dumps_document_v2, migrate_v1_to_v2

    v2_doc = migrate_v1_to_v2(doc) if isinstance(doc, CircuitDocument) else doc
    json_bytes = dumps_document_v2(v2_doc).encode("utf-8")
    compressed = zlib.compress(json_bytes, level=9)
    token = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    return f"{SHARE_URL_PREFIX}{token}"


def decode_circuit_share_url(url: str) -> CircuitDocumentV2:
    """Decode and validate an `ohmbench://c/...` share URL into a `CircuitDocumentV2`."""
    from circuitsim.wiring.serializer import loads_document_v2
    cleaned = str(url).strip()
    if not cleaned.startswith(SHARE_URL_PREFIX):
        raise ShareUrlError(
            f"Invalid share URL scheme; expected prefix {SHARE_URL_PREFIX!r}."
        )
    token = cleaned[len(SHARE_URL_PREFIX) :].strip()
    if not token:
        raise ShareUrlError("Empty share URL payload.")

    pad_len = (-len(token)) % 4
    padded = token + ("=" * pad_len)
    try:
        compressed = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise ShareUrlError(f"Malformed base64url payload in share URL: {exc}") from exc

    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, _MAX_DECOMPRESSED_BYTES + 1)
        if len(raw) > _MAX_DECOMPRESSED_BYTES or decompressor.unconsumed_tail:
            raise ShareUrlError("Share URL payload exceeds maximum allowed decompressed size.")
        json_text = raw.decode("utf-8")
    except ShareUrlError:
        raise
    except Exception as exc:
        raise ShareUrlError(f"Corrupt compressed data in share URL: {exc}") from exc

    try:
        return loads_document_v2(json_text)
    except Exception as exc:
        raise ShareUrlError(f"Invalid schematic document inside share URL: {exc}") from exc
