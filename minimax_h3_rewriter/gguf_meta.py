"""Reading a few keys out of a GGUF header without parsing the whole thing.

``gguf.GGUFReader`` materialises every metadata value the moment it is
constructed, and one of those values is ``tokenizer.ggml.tokens`` -- a quarter of
a million strings turned into Python objects. Measured on this machine:

===========================================  =========  ============
file                                           size      GGUFReader
===========================================  =========  ============
MiniMax-H3-Prompt-Rewriter-LoRA-F16 (no vocab)   3.5 GB       0.03 s
mmproj-Qwen2.5-Omni-3B-Q8_0 (no vocab)           1.4 GB       0.10 s
Qwen2.5-Omni-3B-Q4_K_M                           2.0 GB       3.77 s
Qwen3.6-27B-Q5_K_M                              18.2 GB       6.22 s
gemma-4-E4B-it-qat-UD-Q4_K_XL                    3.9 GB      11.17 s
===========================================  =========  ============

The model list needs six of those keys and the chat template needs one. Ten
files in a model folder cost 31 seconds, paid the first time ComfyUI answers
``/object_info`` -- which is to say, paid as "ComfyUI takes forever to start".

So the header is walked by hand instead. Values that were not asked for are
skipped by seeking past them: a fixed-width array is one seek, and even an array
of strings costs one length read per element with nothing allocated. Parsing
stops as soon as every wanted key has been seen, which for llama.cpp's own
writing order means the tokenizer is usually never reached at all.

Anything unexpected -- a version this does not know, a truncated file, a type id
from the future -- falls back to ``GGUFReader``, so the worst case is the speed
we had before rather than a model that stops being listed.
"""

from __future__ import annotations

import logging
import os
import struct

log = logging.getLogger(__name__)

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_WIDTH = {
    UINT8: 1, INT8: 1,
    UINT16: 2, INT16: 2,
    UINT32: 4, INT32: 4, FLOAT32: 4,
    BOOL: 1,
    UINT64: 8, INT64: 8, FLOAT64: 8,
}

_FORMAT = {
    UINT8: "<B", INT8: "<b",
    UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f",
    BOOL: "<?",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

MAX_STRING = 1 << 20


class MalformedGGUF(ValueError):
    pass


def _exact(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise MalformedGGUF(f"wanted {count} bytes, got {len(data)}")
    return data


def _scalar(handle, kind: int):
    fmt = _FORMAT.get(kind)
    if fmt is None:
        raise MalformedGGUF(f"unknown value type {kind}")
    return struct.unpack(fmt, _exact(handle, _WIDTH[kind]))[0]


def _string(handle) -> str:
    length = struct.unpack("<Q", _exact(handle, 8))[0]
    if length > MAX_STRING:
        raise MalformedGGUF(f"implausible string length {length}")
    return _exact(handle, length).decode("utf-8", errors="replace")


def _skip(handle, kind: int) -> None:
    """Step over one value without building anything from it."""
    if kind in _WIDTH:
        handle.seek(_WIDTH[kind], 1)
        return
    if kind == STRING:
        length = struct.unpack("<Q", _exact(handle, 8))[0]
        handle.seek(length, 1)
        return
    if kind == ARRAY:
        element, count = struct.unpack("<IQ", _exact(handle, 12))
        if element in _WIDTH:
            handle.seek(_WIDTH[element] * count, 1)
            return
        for _ in range(count):
            _skip(handle, element)
        return
    raise MalformedGGUF(f"unknown value type {kind}")


def _value(handle, kind: int):
    if kind == STRING:
        return _string(handle)
    if kind == ARRAY:
        element, count = struct.unpack("<IQ", _exact(handle, 12))
        return [_string(handle) if element == STRING else _scalar(handle, element) for _ in range(count)]
    return _scalar(handle, kind)


DEFAULT_ALIGNMENT = 32
ALIGNMENT_KEY = "general.alignment"


def _check_complete(handle, tensor_count: int, alignment: int, size: int) -> None:
    """Refuse a file whose tensor data does not fit in it.

    ``GGUFReader`` rejected a half-downloaded model as a side effect of mapping
    its tensor data past the end of the file, and dropping it from the model
    list is the right answer -- an interrupted browser download is a plain
    ``.gguf`` with a perfectly valid header. Reading the offsets back is the
    cheap way to keep that behaviour: no allocation, and the table is a few
    hundred fixed-size records.

    The last tensor's *offset* is checked rather than its end, because knowing
    where it ends means knowing the block size of every quantisation format
    llama.cpp has ever shipped. The offset alone catches every truncation that
    happens in practice and needs no such table.
    """
    furthest = 0
    for _ in range(tensor_count):
        _string(handle)  # name
        dimensions = struct.unpack("<I", _exact(handle, 4))[0]
        if dimensions > 8:
            raise MalformedGGUF(f"implausible tensor rank {dimensions}")
        handle.seek(8 * dimensions, 1)
        _kind, offset = struct.unpack("<IQ", _exact(handle, 12))
        furthest = max(furthest, offset)

    alignment = alignment if alignment and alignment > 0 else DEFAULT_ALIGNMENT
    position = handle.tell()
    data_start = position + (-position % alignment)
    if data_start + furthest >= size:
        raise MalformedGGUF(
            f"tensor data starts at {data_start} and reaches at least "
            f"{data_start + furthest}, past the end of a {size}-byte file"
        )


def read_keys(path: str, wanted: tuple[str, ...], probe=None, verify: bool = False) -> dict:
    """Return ``{key: value}`` for the wanted keys present in the header.

    ``probe`` is called with each key already collected and may return further
    keys to look for -- which is how ``<arch>.block_count`` is fetched without
    knowing the architecture until ``general.architecture`` has been read.

    ``verify`` walks the tensor table afterwards and raises when the file is too
    small to hold what it declares. It costs a few hundred fixed-size reads and
    gives up the early exit, so it is for deciding whether to *offer* a file
    rather than for reading a template out of one already chosen.
    """
    found: dict = {}
    remaining = set(wanted)
    alignment = DEFAULT_ALIGNMENT

    with open(path, "rb") as handle:
        if _exact(handle, 4) != MAGIC:
            raise MalformedGGUF("not a GGUF file")
        version = struct.unpack("<I", _exact(handle, 4))[0]
        if version not in SUPPORTED_VERSIONS:
            raise MalformedGGUF(f"GGUF version {version} is not one this parser knows")
        tensor_count, kv_count = struct.unpack("<QQ", _exact(handle, 16))

        for _ in range(kv_count):
            key = _string(handle)
            kind = struct.unpack("<I", _exact(handle, 4))[0]
            if key in remaining:
                found[key] = _value(handle, kind)
                remaining.discard(key)
                if probe is not None:
                    remaining |= {name for name in probe(found) if name not in found}
                if not remaining and not verify:
                    break
            elif verify and key == ALIGNMENT_KEY:
                alignment = int(_value(handle, kind) or DEFAULT_ALIGNMENT)
            else:
                _skip(handle, kind)

        if verify:
            _check_complete(handle, tensor_count, alignment, os.path.getsize(path))

    return found


def keys(path: str, wanted: tuple[str, ...], probe=None, verify: bool = False) -> dict:
    """:func:`read_keys`, falling back to ``gguf.GGUFReader`` if it cannot cope."""
    try:
        return read_keys(path, wanted, probe, verify)
    except MalformedGGUF:
        raise
    except Exception as error:
        log.info(
            "[minimax_h3_rewriter.gguf_meta] fast header read failed on %s (%s), "
            "falling back to GGUFReader", path, error,
        )

    from gguf import GGUFReader

    reader = GGUFReader(path, "r")
    found: dict = {}
    remaining = set(wanted)
    while remaining:
        for name in list(remaining):
            field = reader.fields.get(name)
            remaining.discard(name)
            if field is None:
                continue
            try:
                found[name] = field.contents()
            except Exception:
                log.debug("[minimax_h3_rewriter.gguf_meta] %s unreadable in %s", name, path)
        if probe is not None:
            remaining |= {name for name in probe(found) if name not in found}
    return found

