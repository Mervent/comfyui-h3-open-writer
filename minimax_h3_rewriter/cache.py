"""A run cache for the open writer: identical inputs skip generation entirely.

Rewriting is the one slow step in the node, and it is deterministic whenever
decoding is greedy (or the seed is held). Re-running the same prompt against the
same reference, model and settings therefore reproduces the same text at the cost
of loading a multi-gigabyte model and decoding thousands of tokens for nothing.

This module keeps the answer instead. The cache key is a hash of *everything that
reaches the model* -- the composed system and user messages (which already fold
in the prompt, the reference assets, the fetched guide, the resolution and the
duration) together with the model choice and the decoding parameters. Change any
of them and the key changes, so a stale answer is never returned; leave them all
alone and the node returns the stored text without touching the model.

The copy lands in the ComfyUI user directory next to the fetched guides, so it is
inspectable and removable by hand, and a write that fails never breaks a run --
the worst case is simply generating again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

log = logging.getLogger(__name__)

USER_SUBDIR = "h3_open_writer"
CACHE_DIRNAME = "cache"

#: Bumped when the stored record's shape or the key recipe changes, so old files
#: miss cleanly instead of being read as something they are not.
CACHE_VERSION = 1


def root() -> str:
    """Where cached rewrites live: ``<ComfyUI user>/h3_open_writer/cache``."""
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_user")
    return os.path.join(base, USER_SUBDIR, CACHE_DIRNAME)


def key(messages: list[dict], gen_params: dict) -> str:
    """A stable hash of every input that changes the model's output.

    ``messages`` is the exact system/user payload sent to the model, and
    ``gen_params`` holds the model label and decoding settings that shape how it
    is turned into text. Both are serialised canonically so that the same inputs
    always hash to the same key regardless of dict ordering.
    """
    payload = {
        "version": CACHE_VERSION,
        "messages": messages,
        "gen": gen_params,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def path(cache_key: str) -> str:
    return os.path.join(root(), f"{cache_key}.json")


def load(cache_key: str) -> str | None:
    """Return the stored rewrite for this key, or ``None`` if there is none.

    Anything unexpected -- a missing file, unreadable JSON, a record from an
    older format, an empty body -- is treated as a miss so the caller simply
    regenerates.
    """
    destination = path(cache_key)
    try:
        with open(destination, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        log.warning("[minimax_h3_rewriter.cache.load] ignoring '%s': %s", destination, error)
        return None

    if not isinstance(record, dict) or record.get("version") != CACHE_VERSION:
        return None
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def store(cache_key: str, text: str, meta: dict | None = None) -> None:
    """Write one rewrite into :func:`root`, atomically and best-effort.

    A failure here is logged and swallowed: the run has already produced the
    text, and a cache that cannot be written must never turn a good rewrite into
    an error.
    """
    if not (text or "").strip():
        return

    destination = path(cache_key)
    record = {
        "version": CACHE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_summary": meta or {},
        "text": text,
    }
    part = destination + ".part"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(part, "w", encoding="utf-8") as handle:
            json.dump(
                record,
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(part, destination)
    except OSError as error:
        log.warning("[minimax_h3_rewriter.cache.store] could not write '%s': %s", destination, error)
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        return
    log.info("[minimax_h3_rewriter.cache.store] wrote %s", destination)


def reveal() -> str:
    """Open the cache folder in the desktop file manager."""
    import subprocess
    import sys

    directory = root()
    os.makedirs(directory, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(directory)  # noqa: S606 - opening the user's own folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", directory])
    else:
        subprocess.Popen(["xdg-open", directory])
    return directory
