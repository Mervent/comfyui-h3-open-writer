"""An open MiniMax-H3 Ref2VA writer: system prompt from outside, guide fetched.

This is the guided Ref2VA writer with two things opened up:

- **The system prompt is a node input**, not a constant. Whatever you type into
  ``system_prompt`` becomes the whole system message. Put ``{guide}`` anywhere in
  it and the fetched MiniMax-H3 writing guide is substituted there; leave it out
  and the guide is appended after your text between plain fences.
- **The guide is fetched, never bundled.** ``guide`` picks which official guide
  to download (the full-reference one by default), and it is read fresh on every
  run through the vendored engine's own fetch/cache, so editing the downloaded
  copy takes effect without touching this node.

Everything else -- model resolution, GGUF runtime, context sizing, decoding,
section parsing -- is the vendored engine, called exactly as the built-in Ref2VA
writer calls it.
"""

from __future__ import annotations

import logging

from . import discovery, guide_prompt, guides
from .constants import DURATION_MAX, DURATION_MIN, REF_OUTPUT_FIELDS, RESOLUTIONS
from .fields import split_sections
from .nodes import (
    CATEGORY,
    DEFAULT_OPTIONS,
    OPTIONS_TYPE,
    _ensure_file,
    _gguf_text,
    _report,
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
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "The whole system message. Use {guide} to mark where the fetched "
                            "MiniMax-H3 guide goes; without it the guide is appended after this "
                            "text between fences."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": EXAMPLE_PROMPT,
                        "tooltip": "The original prompt to expand into a Ref2VA description.",
                    },
                ),
                "guide": (
                    list(GUIDE_CHOICES),
                    {
                        "default": "reference",
                        "tooltip": (
                            "Which official guide to fetch and substitute for {guide}: the "
                            "full-reference guide (reference) or the base guide."
                        ),
                    },
                ),
                "model": (
                    writer_choices(),
                    {
                        "tooltip": (
                            "Any GGUF language model. Entries prefixed 'on disk:' are already in "
                            "your ComfyUI model folders; the rest are fetched on first use."
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
            },
            "optional": {
                "reference_assets": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": EXAMPLE_REFERENCE_ASSETS,
                        "tooltip": (
                            "One asset per line — text only. Label them Picture N, Video N or "
                            "Audio N and say what each is for."
                        ),
                    },
                ),
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

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
        reference_assets="",
        options=None,
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        guide_kind = GUIDE_KIND.get(guide, "ref")
        guide_text = guides.text(guide_kind, settings["auto_download"], progress)
        system_content = _compose_system(system_prompt, guide_text)

        user_content = guide_prompt.user_prompt(
            guide_prompt.REF_MODE, prompt, resolution, int(duration), reference_assets or ""
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

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
            adapter_path=None,
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

        _head, sections = split_sections(text, REF_OUTPUT_FIELDS, fallback="detailed_description")
        _report(progress, text, sections, REF_OUTPUT_FIELDS)
        return (text,) + tuple(sections[name] for name in REF_OUTPUT_FIELDS)


NODE_CLASS_MAPPINGS = {"MiniMaxH3OpenWriterRef": MiniMaxH3OpenWriterRef}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3OpenWriterRef": "MiniMax-H3 Open Writer (Ref2VA)",
}
