"""Running a GGUF through a one-shot ``llama-server`` subprocess.

The default GGUF runtime. It needs nothing installed into ComfyUI's Python: see
``llamacpp.py`` for why the wheel is worth avoiding when it is not already
there.

Why the server and not ``llama-completion``, which takes the same arguments and
writes the completion to stdout: speculative decoding lives in the server only.
``--spec-type`` is registered for the server, ``llama-cli`` and the speculative
example, and ``llama-completion`` never reads it. That matters for models that
ship their own multi-token-prediction head (``<arch>.nextn_predict_layers`` in
the header, e.g. Qwen3.8 MTP builds): with ``--spec-type draft-mtp`` the head
drafts a few tokens, the full model verifies them in one batch, and generation
runs roughly 1.5-2x faster with output the target model itself would accept.
A model without such a head runs exactly as before.

The server lives for one generation and is killed afterwards, so the card is
free for video generation the moment the text is done -- VRAM is returned by the
operating system rather than by hoping a deallocator ran, and a llama.cpp crash
takes down a child process instead of ComfyUI and its queue.
``keep_model_loaded = True`` is therefore not honoured here; that is reported
rather than silently ignored.

The prompt is rendered here from the model's own chat template and sent to
``/completion`` as plain text, so ``enable_thinking`` and ``reasoning_effort``
reach the template exactly as they do for the other backends.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import queue
import socket
import threading
import time

import requests

from . import chat_template, devices, llamacpp, runner
from .constants import normalize_seed
from .progress import NodeProgress

log = logging.getLogger(__name__)

PREVIEW_TAIL = 280

ALL_LAYERS = 999

HOST = "127.0.0.1"
HEALTH_TIMEOUT = 2.0
CONNECT_TIMEOUT = 10.0
LOG_TAIL = 40

#: Tokens the MTP head drafts per step. Two is the depth HauhauCS measured for
#: the embedded Qwen3.8 head (2.23x document, 1.60x reasoning on Q8_K_P); deeper
#: drafts are accepted less often once sampling is not greedy.
MTP_DRAFT_TOKENS = 2

ARCHITECTURE_KEY = "general.architecture"
NEXTN_SUFFIX = ".nextn_predict_layers"
TEMPLATE_KEYS = (chat_template.TEMPLATE_KEY, "chat_template")

_METADATA_CACHE: dict[tuple[str, int, int], dict] = {}


def _nextn_key(found: dict) -> tuple[str, ...]:
    architecture = found.get(ARCHITECTURE_KEY)
    return (f"{architecture}{NEXTN_SUFFIX}",) if architecture else ()


def gguf_metadata(model_path: str) -> dict:
    """The chat template and the MTP head count out of a GGUF header.

    Only the header is touched, and the answer is cached per file identity, so
    this costs a stat on every run after the first even for a 30 GB model.
    """
    try:
        stat = os.stat(model_path)
    except OSError as error:
        raise RuntimeError(f"Cannot read '{model_path}': {error}") from error

    key = (os.path.normcase(model_path), stat.st_size, int(stat.st_mtime))
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    from . import gguf_meta

    metadata = gguf_meta.keys(model_path, TEMPLATE_KEYS + (ARCHITECTURE_KEY,), probe=_nextn_key)
    _METADATA_CACHE[key] = metadata
    return metadata


def mtp_layers(metadata: dict) -> int:
    """How many multi-token-prediction layers the model carries; 0 for none."""
    keys = _nextn_key(metadata)
    return int(metadata.get(keys[0]) or 0) if keys else 0


def render_prompt(
    model_path: str,
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
    reasoning_effort: str = "",
) -> str:
    return chat_template.from_metadata(
        gguf_metadata(model_path),
        messages,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


def free_port() -> int:
    """A loopback port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def build_command(
    binary: str,
    model_path: str,
    adapter_path: str | None,
    port: int,
    gpu_layers: int,
    n_ctx: int,
    mtp: bool,
    device: str = devices.AUTO,
) -> list[str]:
    layers = devices.layers_for(device, gpu_layers)
    layers = ALL_LAYERS if layers < 0 else layers
    command = [
        binary,
        "--model", model_path,
        "--host", HOST,
        "--port", str(int(port)),
        *devices.llama_arguments(device),
        "--n-gpu-layers", str(layers),
        "--ctx-size", str(int(n_ctx)),
        # One slot: otherwise the context is split between parallel sequences.
        "--parallel", "1",
        "--no-warmup",
        "--no-webui",
    ]
    if adapter_path:
        command += ["--lora", adapter_path]
    if mtp:
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT_TOKENS)]
    return command


def request_body(
    prompt: str,
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": int(max_new_tokens),
        "seed": normalize_seed(seed),
        "repeat_penalty": float(repetition_penalty),
        "presence_penalty": float(presence_penalty),
        "stream": True,
        "cache_prompt": False,
    }
    if greedy:
        body["temperature"] = 0.0
    else:
        body.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
    return body


def _collect(stream, sink: collections.deque) -> None:
    """Keep the tail of the server's log; reading it also keeps the pipe from filling."""
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", errors="replace").rstrip())
    except Exception:
        log.debug("[minimax_h3_rewriter.server_engine._collect] reader stopped", exc_info=True)


def _tail(logs: collections.deque) -> str:
    return "\n".join(list(logs)[-LOG_TAIL:])


def _wait_ready(process, session, base: str, logs: collections.deque) -> None:
    """Block until ``/health`` answers 200; the server answers 503 while it loads."""
    started = time.monotonic()
    while True:
        if runner.interrupted():
            import comfy.model_management as mm

            raise mm.InterruptProcessingException()
        if process.poll() is not None:
            raise runner.ChildFailed(
                f"llama-server exited with code {process.returncode} while loading the model.\n"
                f"{_tail(logs)}"
            )
        try:
            if session.get(f"{base}/health", timeout=HEALTH_TIMEOUT).status_code == 200:
                return
        except requests.RequestException:
            # Not listening yet: the port opens partway through start-up.
            pass
        waited = time.monotonic() - started
        if waited > runner.FIRST_BYTE_SECONDS:
            raise runner.ChildFailed(
                f"llama-server was not ready after {waited:.0f} s and was stopped, so it could "
                f"not wedge the queue. Last output from it:\n{_tail(logs)}"
            )
        time.sleep(runner.POLL_SECONDS)


def _stream(session, url: str, body: dict, sink: queue.Queue) -> None:
    """Post the request and forward each server-sent event as a dict; ``None`` ends."""
    try:
        with session.post(url, json=body, stream=True, timeout=(CONNECT_TIMEOUT, None)) as response:
            if response.status_code >= 400:
                sink.put({"error": f"HTTP {response.status_code}: {response.text[:2000]}"})
                return
            for line in response.iter_lines():
                if not line:
                    continue
                field, _, payload = line.partition(b":")
                if field not in (b"data", b"error"):
                    continue
                event = json.loads(payload)
                sink.put(event if field == b"data" else {"error": event})
    except Exception as error:
        sink.put({"error": f"{type(error).__name__}: {error}"})
    finally:
        sink.put(None)


def _summary(produced: int, timings: dict) -> str:
    parts = [f"Done · {produced} tokens"]
    speed = timings.get("predicted_per_second")
    if speed:
        parts.append(f"{float(speed):.1f} tok/s")
    drafted = int(timings.get("draft_n") or 0)
    if drafted:
        accepted = int(timings.get("draft_n_accepted") or 0)
        parts.append(f"MTP {accepted}/{drafted} draft tokens accepted")
    return " · ".join(parts)


def generate(
    binary: str,
    model_path: str,
    adapter_path: str | None,
    messages: list[dict[str, str]],
    gpu_layers: int,
    n_ctx: int,
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float = 0.0,
    enable_thinking: bool = False,
    reasoning_effort: str = "",
    device: str = devices.AUTO,
    mtp: bool = True,
    progress: NodeProgress | None = None,
) -> str:
    device = devices.validate(device)
    rendered = render_prompt(
        model_path, messages, enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
    )
    mtp = mtp and mtp_layers(gguf_metadata(model_path)) > 0
    port = free_port()
    command = build_command(binary, model_path, adapter_path, port, gpu_layers, n_ctx, mtp, device)
    body = request_body(
        rendered, seed, greedy, max_new_tokens, temperature, top_p, top_k,
        repetition_penalty, presence_penalty,
    )

    runner.free_comfy_vram(device)
    if progress is not None:
        name = os.path.basename(model_path)
        note = f" + {os.path.basename(adapter_path)}" if adapter_path else " (no adapter)"
        where = "" if device == devices.AUTO else f" on {device}"
        speculative = ", MTP on" if mtp else ""
        progress.set_total(max(int(max_new_tokens), 1))
        progress.text(
            f"Loading {name}{note}\nllama-server{where}, {gpu_layers} GPU layers{speculative}",
            force=True,
        )

    log.info("[minimax_h3_rewriter.server_engine.generate] %s", " ".join(command))
    try:
        process = runner.spawn(command, binary)
    except OSError as error:
        raise RuntimeError(f"Could not start '{binary}': {error}") from error

    logs: collections.deque = collections.deque(maxlen=200)
    for stream in (process.stdout, process.stderr):
        threading.Thread(target=_collect, args=(stream, logs), daemon=True).start()

    session = requests.Session()
    # A system-wide HTTP proxy must not intercept a loopback request.
    session.trust_env = False
    base = f"http://{HOST}:{port}"
    pieces: list[str] = []
    produced = 0
    timings: dict = {}
    try:
        _wait_ready(process, session, base, logs)
        if progress is not None:
            progress.text("Model ready\nprocessing the prompt", force=True)

        events: queue.Queue = queue.Queue()
        threading.Thread(
            target=_stream, args=(session, f"{base}/completion", body, events), daemon=True
        ).start()

        last_event = time.monotonic()
        while True:
            if runner.interrupted():
                import comfy.model_management as mm

                raise mm.InterruptProcessingException()
            try:
                event = events.get(timeout=runner.POLL_SECONDS)
            except queue.Empty:
                limit = runner.STALL_SECONDS if produced else runner.FIRST_BYTE_SECONDS
                waited = time.monotonic() - last_event
                if process.poll() is not None:
                    raise runner.ChildFailed(
                        f"llama-server exited with code {process.returncode} mid-generation.\n"
                        f"{_tail(logs)}"
                    )
                if waited > limit:
                    raise runner.ChildFailed(
                        f"llama-server produced nothing for {waited:.0f} s and was stopped, so it "
                        f"could not wedge the queue. Last output from it:\n{_tail(logs)}"
                    )
                continue
            if event is None:
                break
            last_event = time.monotonic()
            if "error" in event:
                raise runner.ChildFailed(f"llama-server refused the request: {event['error']}\n{_tail(logs)}")
            piece = event.get("content") or ""
            if piece:
                pieces.append(piece)
            produced += len(event.get("tokens") or ()) or (1 if piece else 0)
            if progress is not None and piece:
                tail = "".join(pieces)[-PREVIEW_TAIL:]
                progress.update(produced, f"Generating · {produced}/{max_new_tokens} tokens\n{tail}")
            if event.get("stop"):
                timings = event.get("timings") or {}
                break
    except runner.ChildFailed as error:
        raise RuntimeError(str(error)) from error
    finally:
        session.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        runner.forget(process)

    summary = _summary(produced, timings)
    log.info("[minimax_h3_rewriter.server_engine.generate] %s", summary)
    if progress is not None:
        progress.finish(summary)
    return "".join(pieces).strip()


def rewrite(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    keep_loaded: bool,
    backend: str = "auto",
    auto_download: bool = True,
    progress: NodeProgress | None = None,
    **generation,
) -> str:
    """Fetch the runtime if needed, generate once, and let the process go."""
    binary = llamacpp.ensure_server(backend, auto_download, progress)
    if keep_loaded:
        log.info(
            "[minimax_h3_rewriter.server_engine.rewrite] keep_model_loaded has no effect on the "
            "llama.cpp binary backend: the model leaves with the subprocess"
        )
    return generate(
        binary=binary,
        model_path=model_path,
        adapter_path=adapter_path,
        progress=progress,
        gpu_layers=gpu_layers,
        n_ctx=n_ctx,
        **generation,
    )
