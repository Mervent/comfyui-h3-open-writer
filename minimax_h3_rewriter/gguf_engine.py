"""Running the rewriter from GGUF weights through llama-cpp-python.

The same adapter, converted to GGUF, attaches to a quantised base under
llama.cpp: a ``Q4_K_M`` build of Qwen3.6-27B is 15.7 GB instead of 52 GB, and
smaller quants go lower still. llama-cpp-python is an optional dependency —
absent it, this backend is simply unavailable and the node says so.

The prompt is rendered from the GGUF's own chat template rather than through
llama-cpp-python's chat formatter, because the reference implementation passes
``enable_thinking=False`` and the formatter has no way to forward it.

Attaching the adapter takes two different shapes depending on the build, and the
difference has to be settled by looking rather than by trying. The long-standing
spelling is ``Llama(lora_path=...)``, applied during construction. Some builds --
JamePeng's CUDA fork from 0.3.47 onwards, for one -- have replaced it with a
registry: ``load_lora(name, path)`` after construction, then
``active_loras=[{"name": ..., "scale": ...}]`` on the call that generates. What
makes this dangerous rather than merely different is that ``Llama.__init__``
takes a ``**kwargs`` it never reads, in every build including upstream's, so an
argument it does not know is accepted and discarded. A ``lora_path`` handed to a
build that dropped it raises nothing at all: the node loads, logs the adapter,
and answers from the plain base model. Hence :func:`_lora_api`, which asks the
signature, and :func:`_register_adapter`, which reads the registry back.
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
import sys
import threading

from . import chat_template, devices
from .constants import install_command, normalize_seed
from .progress import NodeProgress

log = logging.getLogger(__name__)

_STATE: dict = {"key": None, "llama": None, "lora": None}
_LOCK = threading.RLock()

LORA_MODERN = "modern"
LORA_LEGACY = "legacy"

PREVIEW_TAIL = 280
RELEASES_URL = "https://github.com/abetlen/llama-cpp-python/releases"
WHEEL_RELEASE = "v0.3.34-vulkan"
WHEEL_FILES = {
    "win32": "llama_cpp_python-0.3.34-py3-none-win_amd64.whl",
    "linux": "llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
}


def wheel_url() -> str:
    """The prebuilt Vulkan wheel for this platform, or "" if there is none."""
    name = WHEEL_FILES.get(sys.platform)
    return f"{RELEASES_URL}/download/{WHEEL_RELEASE}/{name}" if name else ""


def _install_hint() -> str:
    head = (
        "GGUF models need llama-cpp-python, which is not installed in this Python "
        "environment.\n"
    )
    tail = (
        "\nInstalling it is optional: without it the node runs GGUF models through the "
        "official llama.cpp binaries instead, which it fetches on first use. The wheel is "
        "worth having only if you want the model to stay resident between runs."
    )
    url = wheel_url()
    if not url:
        return (
            head
            + f"    {install_command('llama-cpp-python')}\n"
            + f"Prebuilt wheels for other platforms are at {RELEASES_URL}."
            + tail
        )
    return (
        head
        + "Use the Vulkan build. It runs on NVIDIA, AMD and Intel GPUs alike, and "
        "needs no match between the wheel's CUDA version and your driver:\n"
        f"    {install_command(url)}\n"
        "Then restart ComfyUI. The CUDA wheels at that same releases page are "
        "faster when they fit, but cu130 and earlier are compiled with AVX-512 and "
        "die with 0xC000001D on consumer Intel 12th-14th generation CPUs, while "
        "cu132 ships PTX that a driver older than CUDA 13.2 refuses to compile."
        + tail
    )


INSTALL_HINT = _install_hint()


def available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


def _llama_cpp():
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError(INSTALL_HINT) from error
    return llama_cpp


def _free_comfy_vram(device: str = devices.AUTO) -> None:
    from . import runner

    runner.free_comfy_vram(device)


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except Exception:
        return False


def _lora_api(llama_cpp) -> str:
    """Which adapter API this build offers: modern, legacy, or none at all.

    Asked of the signature rather than of a ``try``/``except``, because the way
    this fails is silence: the constructor swallows what it does not recognise.
    """
    llama = getattr(llama_cpp, "Llama", None)
    if llama is None:
        return ""
    if callable(getattr(llama, "load_lora", None)):
        return LORA_MODERN
    try:
        parameters = inspect.signature(llama.__init__).parameters
    except (TypeError, ValueError):
        return LORA_LEGACY
    return LORA_LEGACY if "lora_path" in parameters else ""


def _register_adapter(llama, adapter_path: str) -> str:
    """Load the adapter into a modern build, and confirm it really landed.

    ``Llama.eval`` looks each adapter up by name and skips a miss with a warning
    it prints only when ``verbose`` -- which this backend turns off, since
    llama.cpp's loader output is not what a ComfyUI console is for. Reading the
    registry back is what stands in for that warning: without it a refused file
    would leave the base model answering under the rewriter's name, which is the
    whole failure this branch exists to prevent.
    """
    name = os.path.splitext(os.path.basename(adapter_path))[0] or "adapter"
    llama.load_lora(name, adapter_path)

    lookup = getattr(llama, "list_loras", None)
    registered = list(lookup()) if callable(lookup) else [name]
    if name not in registered:
        raise RuntimeError(
            f"llama-cpp-python took the adapter '{adapter_path}' but did not register it "
            f"as '{name}' (registry: {registered or 'empty'}). Refusing to generate, "
            "because the model would answer as though it had never been fine-tuned."
        )

    log.info(
        "[minimax_h3_rewriter.gguf.load] adapter registered as %r (%d loaded)",
        name,
        len(registered),
    )
    return name


def _active_loras(llama):
    """The ``active_loras`` argument for this instance, or ``None``.

    Only the modern API needs it -- the legacy one applied the adapter while the
    model was being built. The identity check keeps a stale name from a previous
    load off an instance that was created without one.
    """
    with _LOCK:
        name = _STATE.get("lora") if _STATE.get("llama") is llama else None
    return [{"name": name, "scale": 1.0}] if name else None


def _release_adapters(llama) -> None:
    """Free the adapters while the model they point into is still alive.

    ``LlamaModel.close`` frees the base model first and walks its adapter
    registry afterwards, so ``llama_adapter_lora_free`` runs against memory that
    is already gone -- an access violation that takes ComfyUI's worker thread
    with it, on every unload. Doing it here hands ``close`` an empty registry.
    Nothing is freed twice: the adapter's own ``free`` guards on its pointer, so
    the later passes are no-ops.
    """
    release = getattr(llama, "unload_all_loras", None)
    if not callable(release):
        return
    try:
        release()
    except Exception:
        log.debug("[minimax_h3_rewriter.gguf.unload] adapter release failed", exc_info=True)


def load(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    device: str = devices.AUTO,
    progress: NodeProgress | None = None,
):
    """Return a cached ``Llama`` for this combination of files and placement.

    The device is part of the key: without it, switching cards would hand back
    the instance still resident on the old one.
    """
    device = devices.validate(device)
    key = (
        os.path.normcase(model_path),
        os.path.normcase(adapter_path or ""),
        int(gpu_layers),
        int(n_ctx),
        device,
    )

    with _LOCK:
        if _STATE["key"] == key and _STATE["llama"] is not None:
            return _STATE["llama"]

        unload()
        _free_comfy_vram(device)

        llama_cpp = _llama_cpp()

        if progress is not None:
            progress.set_total(1000)
            name = os.path.basename(model_path)
            adapter_note = f" + {os.path.basename(adapter_path)}" if adapter_path else " (no adapter)"
            where = "" if device == devices.AUTO else f" on {device}"
            progress.ratio(
                0.05, f"Loading {name}{adapter_note}\nllama.cpp{where}, {gpu_layers} GPU layers"
            )

        kwargs = {
            "model_path": model_path,
            "n_gpu_layers": devices.layers_for(device, gpu_layers),
            "n_ctx": int(n_ctx),
            "verbose": False,
        }
        kwargs.update(devices.llama_cpp_kwargs(device))

        api = _lora_api(llama_cpp) if adapter_path else ""
        if adapter_path and not api:
            raise RuntimeError(
                "This llama-cpp-python build offers no way to attach a LoRA: 'lora_path' "
                "is gone from Llama() and there is no load_lora() to put it back. Set "
                "gguf_runtime to 'llama.cpp' to run the official binaries, which pass "
                "--lora and were never affected, or turn 'use_lora' off to run the base "
                "model on its own."
            )
        if api == LORA_LEGACY:
            # Applied during construction, and this spelling reports its own
            # failures: the constructor raises when the adapter will not load.
            kwargs["lora_path"] = adapter_path

        llama = llama_cpp.Llama(**kwargs)
        _STATE.update(key=key, llama=llama, lora=None)
        if api == LORA_MODERN:
            try:
                _STATE["lora"] = _register_adapter(llama, adapter_path)
            except Exception:
                # The weights are already in VRAM; refusing must not strand them.
                unload()
                raise
        if progress is not None:
            progress.ratio(1.0, "Model ready")
        return llama


def unload() -> None:
    with _LOCK:
        llama = _STATE.get("llama")
        _STATE.update(key=None, llama=None, lora=None)
    if llama is not None:
        _release_adapters(llama)
        close = getattr(llama, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("[minimax_h3_rewriter.gguf.unload] close failed", exc_info=True)
        del llama
    gc.collect()


def is_loaded() -> bool:
    return _STATE["llama"] is not None


def _render(llama, messages: list[dict[str, str]]) -> str:
    metadata = dict(getattr(llama, "metadata", {}) or {})
    return chat_template.from_metadata(metadata, messages, enable_thinking=False)


def generate(
    llama,
    messages: list[dict[str, str]],
    seed: int,
    greedy: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress: NodeProgress | None = None,
) -> str:
    rendered = _render(llama, messages)

    call_kwargs = {
        "max_tokens": int(max_new_tokens),
        "repeat_penalty": float(repetition_penalty),
        "stream": True,
        "seed": normalize_seed(seed),
    }
    active = _active_loras(llama)
    if active is not None:
        call_kwargs["active_loras"] = active
    if greedy:
        call_kwargs["temperature"] = 0.0
    else:
        call_kwargs.update(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

    if progress is not None:
        progress.set_total(max(int(max_new_tokens), 1))
        progress.update(0, "Generating\n0 tokens")

    try:
        stream = llama(rendered, **call_kwargs)
    except TypeError:
        # Old builds have no 'seed'. Retry without it, and only it -- dropping
        # 'active_loras' would run the base model as though the adapter were
        # attached, which is exactly the silence this backend guards against.
        call_kwargs.pop("seed", None)
        stream = llama(rendered, **call_kwargs)

    pieces: list[str] = []
    produced = 0
    interrupted = False
    for chunk in stream:
        try:
            piece = chunk["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
        if not piece:
            continue
        pieces.append(piece)
        produced += 1
        if progress is not None:
            tail = "".join(pieces)[-PREVIEW_TAIL:]
            progress.update(produced, f"Generating · {produced}/{max_new_tokens} tokens\n{tail}")
        if _interrupted():
            interrupted = True
            break

    if interrupted:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("[minimax_h3_rewriter.gguf.generate] stream close failed", exc_info=True)
        import comfy.model_management as mm

        raise mm.InterruptProcessingException()

    if progress is not None:
        progress.finish(f"Done · {produced} tokens")
    return "".join(pieces).strip()


def rewrite(
    model_path: str,
    adapter_path: str | None,
    gpu_layers: int,
    n_ctx: int,
    keep_loaded: bool,
    device: str = devices.AUTO,
    progress: NodeProgress | None = None,
    **generation,
) -> str:
    """Load (or reuse) the GGUF rewriter, generate once, release unless kept."""
    llama = load(model_path, adapter_path, gpu_layers, n_ctx, device, progress)
    try:
        return generate(llama, progress=progress, **generation)
    finally:
        del llama
        if not keep_loaded:
            unload()
