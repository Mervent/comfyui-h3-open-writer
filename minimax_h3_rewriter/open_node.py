"""Open MiniMax-H3 writers: system prompt from outside, guide fetched.

Two nodes, the guided Ref2VA and FL2VA writers with two things opened up:

- **The system prompt is a node input**, not a constant. Whatever you type into
  ``system_prompt`` becomes the whole system message. Put ``{guide}`` anywhere in
  it and the fetched MiniMax-H3 writing guide is substituted there; leave it out
  and the guide is appended after your text between plain fences. The FL2VA
  node also fills in ``{seconds}``: its alignment line has to carry the duration
  to two decimal places, and the duration is a node input.
- **The guide is fetched, never bundled.** The Ref2VA node's ``guide`` picks
  which official guide to download (the full-reference one by default); FL2VA is
  always served by the base guide. It is read fresh on every run through the
  vendored engine's own fetch/cache, so editing the downloaded copy takes effect
  without touching either node.

Everything else -- model resolution, GGUF runtime, context sizing, decoding,
section parsing -- is the vendored engine, called exactly as the built-in
guided writers call it.
"""

from __future__ import annotations

import logging

from . import cache, discovery, guide_prompt, guides
from .catalog import FORMAT_GGUF
from .constants import DURATION_MAX, DURATION_MIN, OUTPUT_FIELDS, REF_OUTPUT_FIELDS, RESOLUTIONS
from .fields import split_sections
from .nodes import (
    CATEGORY,
    DEFAULT_OPTIONS,
    OPTIONS_TYPE,
    _ensure_file,
    _gguf_text,
    _report,
    _resolve_adapter,
    _resolve_writer_choice,
    writer_choices,
)
from .progress import NodeProgress

log = logging.getLogger(__name__)

GUIDE_CHOICES = ("reference", "base")
GUIDE_KIND = {"reference": "ref", "base": "base"}

DEFAULT_SYSTEM_PROMPT = """You are a professional prompt rewriter for MiniMax-H3 joint audio-video generation.
Rewrite the user's original prompt into one full-reference (Ref2VA) description
that follows the writing guide below to the letter.

===== BEGIN MINIMAX-H3 WRITING GUIDE =====
{guide}
===== END MINIMAX-H3 WRITING GUIDE =====

The requested task is Ref2VA — full-reference generation: reference assets define subjects, frames, structure or audio that the target video reuses.

Output contract:
- Return only these six fields, in this exact order, each introduced by its own
  name followed by a colon:
  subject_definitions: ...
  summary: ...
  retention_analysis: ...
  detailed_description: ...
  overall_soundscape: ...
  non_diegetic_music: ...
- Every asset listed under reference_assets gets a label in subject_definitions and
  a line in retention_analysis. Do not invent assets that are not listed there.
- summary starts with its square-bracketed task-type prefix.
- detailed_description opens with one or two sentences of overall style before
  [Shot 1], and cites each label where its role actually applies.
- Compose the scene for the requested aspect ratio, and fit the number, timing and
  pacing of the shots to the requested duration.
- Write everything in English. Dialogue and lyrics inside <d> and text visible on
  screen keep their original wording and punctuation.
- Do not add explanations, notes, headings, Markdown fences, or any field that is
  not listed above. Do not restate the guide."""

FL2VA_SYSTEM_PROMPT = """You are a professional prompt rewriter for MiniMax-H3 joint audio-video generation.
Rewrite the user's original prompt into one coherent, production-ready
first-and-last-frame (FL2VA) description that follows the writing guide below to
the letter.

===== BEGIN MINIMAX-H3 WRITING GUIDE =====
{guide}
===== END MINIMAX-H3 WRITING GUIDE =====

The requested task is FL2VA — first-and-last-frame to audio-video: Picture 1 opens the video, Picture 2 closes it, and the body is the path between them.

Output contract:
- Start the answer with exactly this line, then one blank line, then the fields:
    How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the {seconds}-second mark of the target video.
  Replace N with the number of the actual final shot. Change nothing else in it.
- Then return only these three fields, in this exact order, each introduced by its
  own name followed by a colon:
  integrated_multimodal_description: ...
  overall_soundscape: ...
  non_diegetic_music: ...
- Shot 1 opens on exactly what Picture 1 shows, and the final shot lands on exactly
  what Picture 2 shows. Take both from reference_material; never contradict it, and
  do not invent a third picture.
- The shots in between are a plausible, continuous path from the first frame to the
  last: who moves where, what changes, and what the camera does to get there.
- Compose the scene for the requested aspect ratio, and fit the number, timing and
  pacing of the shots to the requested duration.
- When original_prompt gives timings, such as [1.5-4.0s], keep every one of them and
  do not turn them into cuts. Stay in the current shot and open each timed beat with
  its start time in the guide's MM:SS.mmm form, e.g. "At 00:01.500, she ...", in the
  user's order. A beat is not a shot: no new [Shot N], no "cuts to" or transition
  wording, and [Shot 1] itself still carries no timestamp. Pace each beat so it is
  done by the end of its range. Cut only where the user explicitly asks for a cut.
- Write everything in English. Dialogue and lyrics inside <d> and text visible on
  screen keep their original wording and punctuation.
- Use N/A for a sound field only in the cases the guide allows it.
- Do not add explanations, notes, headings, Markdown fences, or any field that is
  not listed above. Do not restate the guide."""

GUIDE_OPEN = "===== BEGIN MINIMAX-H3 WRITING GUIDE ====="
GUIDE_CLOSE = "===== END MINIMAX-H3 WRITING GUIDE ====="

EXAMPLE_PROMPT = (
    "A quiet, cinematic moment. The young woman from Picture 1 sits by the rainy cafe "
    "window and remembers the night walk from Video 1. Warm interior light, melancholic "
    "mood; she murmurs one short line to herself in her referenced voice."
)

EXAMPLE_REFERENCE_ASSETS = (
    "Picture 1: young woman, long dark hair, blue cardigan, seated by a window\n"
    "Video 1: source clip being edited — handheld walk down a night street\n"
    "Audio 1: voice-timbre reference for the woman"
)

FL2VA_EXAMPLE_PROMPT = (
    "She spots him across the snowy platform, hesitates, then walks over as the train "
    "pulls in. One quiet line of greeting; distant station announcements."
)

FL2VA_EXAMPLE_REFERENCE = (
    "Picture 1: first frame — young woman in a red coat alone on a snowy train platform "
    "at dusk, wide shot\n"
    "Picture 2: last frame — the same woman face to face with a man in a grey scarf, "
    "close two-shot, train lights behind them"
)


def _compose_system(system_prompt: str, guide_text: str) -> str:
    """Put the guide where the user asked, or after their text if they did not."""
    system_prompt = (system_prompt or "").strip()
    guide_text = (guide_text or "").strip()
    if "{guide}" in system_prompt:
        return system_prompt.replace("{guide}", guide_text)
    if not guide_text:
        return system_prompt
    if not system_prompt:
        return guide_text
    return f"{system_prompt}\n\n{GUIDE_OPEN}\n{guide_text}\n{GUIDE_CLOSE}"


def _inputs(
    system_default: str,
    system_tooltip: str,
    prompt_default: str,
    prompt_tooltip: str,
    guide_input: dict | None,
    references: tuple[str, str, str],
) -> dict:
    """The widgets both open writers share, in the order saved workflows rely on.

    ComfyUI stores widget values by position, so ``guide`` -- present on the
    Ref2VA node only -- has to stay right after ``prompt`` there.
    """
    required = {
        "system_prompt": (
            "STRING",
            {"multiline": True, "default": system_default, "tooltip": system_tooltip},
        ),
        "prompt": (
            "STRING",
            {"multiline": True, "default": prompt_default, "tooltip": prompt_tooltip},
        ),
    }
    if guide_input is not None:
        required["guide"] = guide_input
    required.update({
        "model": (
            writer_choices(),
            {
                "tooltip": (
                    "Any GGUF language model. Entries prefixed 'on disk:' are already in "
                    "your ComfyUI model folders; the rest are fetched on first use. "
                    "With 'use_lora' on (the options node's default) the prompt-rewriter "
                    "LoRA is applied, so the base must match it (Qwen3.6-27B); turn "
                    "'use_lora' off to run any other GGUF plain."
                ),
            },
        ),
        "resolution": (
            list(RESOLUTIONS),
            {"default": "16:9", "tooltip": "Target aspect ratio the rewrite is composed for."},
        ),
        "duration": (
            "INT",
            {
                "default": 10,
                "min": DURATION_MIN,
                "max": DURATION_MAX,
                "step": 1,
                "tooltip": "Target clip length in seconds; drives shot count and pacing.",
            },
        ),
        "greedy": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Deterministic decoding. Turn off (and randomize seed) for variation.",
            },
        ),
        "use_cache": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Reuse a stored rewrite when the prompt, reference, model and every "
                    "setting are unchanged, instead of running the model again. Turn off to "
                    "always regenerate. The 'Open cache folder' button shows where the "
                    "rewrites are kept."
                ),
            },
        ),
        "seed": (
            "INT",
            {
                "default": 42,
                "min": 0,
                "max": 0xFFFFFFFF,
                "control_after_generate": True,
            },
        ),
        "keep_model_loaded": (
            "BOOLEAN",
            {"default": False, "tooltip": "Keep the writer in VRAM after the rewrite."},
        ),
    })
    name, default, tooltip = references
    return {
        "required": required,
        "optional": {
            name: ("STRING", {"multiline": True, "default": default, "tooltip": tooltip}),
            "options": (OPTIONS_TYPE,),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }


def _write(
    mode: str,
    fields: tuple[str, ...],
    fallback: str,
    guide: str,
    system_content: str,
    prompt: str,
    references: str,
    model: str,
    resolution: str,
    duration: int,
    greedy: bool,
    seed: int,
    keep_model_loaded: bool,
    use_cache: bool,
    settings: dict,
    progress: NodeProgress,
) -> tuple[str, ...]:
    """Run one open writer and return ``(whole text, *fields)``."""
    user_content = guide_prompt.user_prompt(mode, prompt, resolution, int(duration), references or "")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    gen_params = {
        "model": model,
        "greedy": bool(greedy),
        "seed": int(seed),
        "max_new_tokens": int(settings["max_new_tokens"]),
        "temperature": float(settings["temperature"]),
        "top_p": float(settings["top_p"]),
        "top_k": int(settings["top_k"]),
        "repetition_penalty": float(settings["repetition_penalty"]),
        "presence_penalty": float(settings["presence_penalty"]),
        "use_lora": bool(settings["use_lora"]),
        "adapter": settings["adapter"] if settings["use_lora"] else "",
    }
    cache_key = cache.key(messages, gen_params) if use_cache else ""

    if cache_key:
        cached = cache.load(cache_key)
        if cached is not None:
            _head, sections = split_sections(cached, fields, fallback=fallback)
            progress.text(
                "Reused a cached rewrite; the model was not run.\n\n"
                + (cached[-2000:] if cached else "(empty rewrite)"),
                force=True,
            )
            log.info("[minimax_h3_rewriter.open_node] cache hit %s", cache_key)
            return (cached,) + tuple(sections[name] for name in fields)

    choice = _resolve_writer_choice(model)
    if choice.local:
        model_path = choice.reference
    else:
        model_path = _ensure_file(
            choice.reference, choice.file, "Writer model", settings["auto_download"], progress
        )
    if discovery.gguf_header(model_path)["kind"] == "adapter":
        raise RuntimeError(
            f"'{model_path}' is a LoRA adapter, not a model that can be run on its own. "
            f"Pick a base model from the list."
        )

    adapter_path = None
    if settings["use_lora"]:
        problem = discovery.gguf_problem(model_path)
        if problem:
            raise RuntimeError(
                "This GGUF cannot run the prompt-rewriter LoRA.\n  - "
                + problem
                + "\nTurn 'use_lora' off to run it as a plain model anyway."
            )
        adapter_path = _resolve_adapter(
            FORMAT_GGUF, settings["adapter"], settings["auto_download"], progress
        )

    max_new_tokens = int(settings["max_new_tokens"])
    n_ctx = int(settings["n_ctx"])
    needed = guide_prompt.context_needed(messages, max_new_tokens)
    if needed > n_ctx:
        progress.text(
            f"the guide needs a {needed}-token context, raising n_ctx from {n_ctx}",
            force=True,
        )
        n_ctx = needed

    text = _gguf_text(
        settings,
        model_path=model_path,
        adapter_path=adapter_path,
        gpu_layers=int(settings["gpu_layers"]),
        n_ctx=n_ctx,
        keep_loaded=keep_model_loaded,
        device=settings["device"],
        progress=progress,
        messages=messages,
        seed=int(seed),
        greedy=greedy,
        max_new_tokens=max_new_tokens,
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        top_k=int(settings["top_k"]),
        repetition_penalty=float(settings["repetition_penalty"]),
    )

    _head, sections = split_sections(text, fields, fallback=fallback)
    _report(progress, text, sections, fields)

    if cache_key:
        cache.store(
            cache_key,
            text=text,
            meta={
                "model": model,
                "guide": guide,
                "resolution": resolution,
                "duration": int(duration),
                "greedy": bool(greedy),
                "seed": int(seed),
                "use_lora": bool(settings["use_lora"]),
                "prompt_preview": (prompt or "").strip()[:200],
            },
        )

    return (text,) + tuple(sections[name] for name in fields)


def _settings(options) -> dict:
    settings = dict(DEFAULT_OPTIONS)
    if options:
        settings.update(options)
    return settings


class MiniMaxH3OpenWriterRef:
    """Ref2VA writer whose system prompt is supplied and whose guide is fetched."""

    DESCRIPTION = (
        "A full-reference (Ref2VA) MiniMax-H3 writer with the system prompt exposed as an "
        "input and the official writing guide fetched at run time. Put {guide} in the system "
        "prompt to place the guide, or leave it out to append the guide after your text. "
        "Runs on any instruction-following GGUF through the vendored engine."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _inputs(
            system_default=DEFAULT_SYSTEM_PROMPT,
            system_tooltip=(
                "The whole system message. Use {guide} to mark where the fetched "
                "MiniMax-H3 guide goes; without it the guide is appended after this "
                "text between fences."
            ),
            prompt_default=EXAMPLE_PROMPT,
            prompt_tooltip="The original prompt to expand into a Ref2VA description.",
            guide_input=(
                list(GUIDE_CHOICES),
                {
                    "default": "reference",
                    "tooltip": (
                        "Which official guide to fetch and substitute for {guide}: the "
                        "full-reference guide (reference) or the base guide."
                    ),
                },
            ),
            references=(
                "reference_assets",
                EXAMPLE_REFERENCE_ASSETS,
                "One asset per line — text only. Label them Picture N, Video N or "
                "Audio N and say what each is for.",
            ),
        )

    RETURN_TYPES = ("STRING",) * (1 + len(REF_OUTPUT_FIELDS))
    RETURN_NAMES = ("rewritten_prompt",) + REF_OUTPUT_FIELDS
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        system_prompt,
        prompt,
        guide,
        model,
        resolution,
        duration,
        greedy,
        seed,
        keep_model_loaded,
        use_cache=True,
        reference_assets="",
        options=None,
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        settings = _settings(options)
        progress = NodeProgress(unique_id)
        guide_text = guides.text(GUIDE_KIND.get(guide, "ref"), settings["auto_download"], progress)
        return _write(
            guide_prompt.REF_MODE, REF_OUTPUT_FIELDS, "detailed_description", guide,
            _compose_system(system_prompt, guide_text), prompt, reference_assets,
            model, resolution, duration, greedy, seed, keep_model_loaded, use_cache,
            settings, progress,
        )


class MiniMaxH3OpenWriterFL2VA:
    """FL2VA writer whose system prompt is supplied and whose guide is fetched."""

    MODE = "FL2VA"

    DESCRIPTION = (
        "A first-and-last-frame (FL2VA) MiniMax-H3 writer with the system prompt exposed as an "
        "input and the official base writing guide fetched at run time. Put {guide} in the "
        "system prompt to place the guide, or leave it out to append it after your text; "
        "{seconds} becomes the duration for the alignment line. It reads text, not pixels: "
        "describe both frames in 'reference_material'. Runs on any instruction-following GGUF "
        "through the vendored engine."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _inputs(
            system_default=FL2VA_SYSTEM_PROMPT,
            system_tooltip=(
                "The whole system message. Use {guide} to mark where the fetched "
                "MiniMax-H3 base guide goes; without it the guide is appended after this "
                "text between fences. {seconds} is replaced by the duration to two "
                "decimals, e.g. 10.00, for the alignment line."
            ),
            prompt_default=FL2VA_EXAMPLE_PROMPT,
            prompt_tooltip="What happens between the first and the last frame.",
            guide_input=None,
            references=(
                "reference_material",
                FL2VA_EXAMPLE_REFERENCE,
                "This node reads text, not pixels. Describe what the first frame "
                "(Picture 1) and the last frame (Picture 2) show, one per line — by hand, "
                "or from a captioner node.",
            ),
        )

    RETURN_TYPES = ("STRING",) * (1 + len(OUTPUT_FIELDS))
    RETURN_NAMES = ("rewritten_prompt",) + OUTPUT_FIELDS
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        system_prompt,
        prompt,
        model,
        resolution,
        duration,
        greedy,
        seed,
        keep_model_loaded,
        use_cache=True,
        reference_material="",
        options=None,
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        settings = _settings(options)
        progress = NodeProgress(unique_id)
        guide_kind = guide_prompt.GUIDE_FOR_MODE[self.MODE]
        guide_text = guides.text(guide_kind, settings["auto_download"], progress)
        # Before the guide goes in, so a '{seconds}' inside the guide is left alone.
        system_prompt = (system_prompt or "").replace("{seconds}", f"{float(duration):.2f}")
        return _write(
            self.MODE, OUTPUT_FIELDS, "", guide_kind,
            _compose_system(system_prompt, guide_text), prompt, reference_material,
            model, resolution, duration, greedy, seed, keep_model_loaded, use_cache,
            settings, progress,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3OpenWriterRef": MiniMaxH3OpenWriterRef,
    "MiniMaxH3OpenWriterFL2VA": MiniMaxH3OpenWriterFL2VA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3OpenWriterRef": "MiniMax-H3 Open Writer (Ref2VA)",
    "MiniMaxH3OpenWriterFL2VA": "MiniMax-H3 Open Writer (FL2VA)",
}
