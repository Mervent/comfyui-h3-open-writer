"""A creativity pass over a prompt, driven by inline ``<imagine>`` markers.

This node does one narrow thing: it reimagines the spans a user marks and
leaves everything else untouched. A span is written inline in the prompt as::

    <imagine str=3>a quiet street</imagine>

and the model rewrites only the text inside it, splicing the result back in
wrapped in ``*asterisks*`` so it is obvious what changed. ``str`` is the
imagination strength for that one span, from 0 (barely touch it) to 9 (the most
imaginative reading the words can bear); a span written as a bare ``<imagine>``
uses the node's ``imagination`` default instead.

It is a pre-processor, not a writer: one ``creative_prompt`` string comes out,
which you then feed into any of the MiniMax-H3 writer nodes. The whole prompt
goes through the model in a single pass, exactly as the guided writers work --
system prompt explaining the contract, user prompt carrying the tagged text --
so there is no per-span round trip and the sentences keep their context.
"""

from __future__ import annotations

import logging
import re

from . import discovery, guide_prompt
from .nodes import (
    BYPASS_TOOLTIP,
    CATEGORY,
    DEFAULT_OPTIONS,
    OPTIONS_TYPE,
    _bypassed,
    _ensure_file,
    _gguf_text,
    _resolve_writer_choice,
    writer_choices,
)
from .progress import NodeProgress

log = logging.getLogger(__name__)

IMAGINE_RE = re.compile(
    r"<imagine(?:\s+str=(\d+))?\s*>(.*?)</imagine>",
    re.DOTALL | re.IGNORECASE,
)

LEVEL_MIN = 0
LEVEL_MAX = 9

DEFAULT_SYSTEM_PROMPT = """You are a creative prompt embellisher. You receive a prompt that may contain
one or more marked spans written as:

    <imagine str=N>TEXT</imagine>

Your only job is to reimagine the TEXT inside each such span more vividly and
creatively, and to leave every character outside the spans exactly as it is.

Rules:
- Rewrite only the text inside <imagine>...</imagine>. Never change, reorder,
  add, or drop anything outside a span.
- Replace each whole span, tags included, with your reimagined text wrapped in
  single asterisks, like *this*. The <imagine> and </imagine> tags must never
  appear in your output.
- str=N is the imagination strength for that span, from 0 to 9. A span written
  without str= uses the default level of {level}.
- Keep the reimagined text grammatically consistent with the surrounding words,
  so the sentence still reads naturally once the span is replaced.
- Do not answer, explain, comment, or add headings. Return only the rewritten
  prompt.

Imagination-strength scale:
  0    leave it essentially as written; touch only obvious wording
  1-2  light polish: a stronger adjective or two, same meaning
  3-4  vivid: concrete sensory detail and richer verbs, still literal
  5-6  inventive: add mood, texture, and unexpected but fitting imagery
  7-8  bold: surprising metaphors and striking, cinematic detail
  9    surreal: the most imaginative reading the words can bear, while still
       describing the same subject"""

EXAMPLE_PROMPT = (
    "A woman walks down <imagine str=3>a quiet street</imagine> at night, "
    "and the camera lingers on <imagine str=8>her face</imagine>."
)

LEVEL_LINE = "\n\nSpans written without str= use the default imagination level of {level}."


def _compose_system(system_prompt: str, level: int) -> str:
    """Fill the default level into the system prompt the user supplied.

    ``{level}`` is where the imagination default belongs, so it is substituted
    there when present. A system prompt that never mentions it -- one the user
    rewrote from scratch -- gets the default stated in one appended line rather
    than silently, so a bare ``<imagine>`` still has a defined strength.
    """
    system_prompt = (system_prompt or "").strip()
    if "{level}" in system_prompt:
        return system_prompt.replace("{level}", str(level))
    return system_prompt + LEVEL_LINE.replace("{level}", str(level))


def _outside_segments(prompt: str) -> list[str]:
    """The literal text between the tags -- everything that must survive verbatim."""
    segments: list[str] = []
    last = 0
    for match in IMAGINE_RE.finditer(prompt):
        segments.append(prompt[last : match.start()])
        last = match.end()
    segments.append(prompt[last:])
    return segments


def _preservation_note(prompt: str, output: str) -> str:
    """Warn -- without failing -- when the model dropped the untouched text.

    A single-pass rewrite trusts the model to copy everything outside the spans
    byte for byte, and a small one sometimes paraphrases it instead. The literal
    segments long enough to be distinctive are checked against the output; if
    several have gone missing, the run still returns, but the node says the text
    outside the spans may have drifted so it is not mistaken for a clean pass.
    """
    anchors = [segment.strip() for segment in _outside_segments(prompt)]
    anchors = [anchor for anchor in anchors if len(anchor) >= 12]
    if not anchors:
        return ""
    missing = [anchor for anchor in anchors if anchor not in output]
    if len(missing) * 2 <= len(anchors):
        return ""
    log.warning(
        "[minimax_h3_rewriter.creative_node] %d/%d literal segments missing from the "
        "rewrite; the model may have altered text outside the <imagine> spans",
        len(missing),
        len(anchors),
    )
    return (
        f"⚠ {len(missing)} of {len(anchors)} untouched segment(s) are not in the output — "
        f"the model may have changed text outside the <imagine> spans. Lower the temperature "
        f"or try a larger model.\n\n"
    )


class MiniMaxH3CreativeImaginer:
    """Reimagine the ``<imagine>``-marked spans of a prompt, leaving the rest alone."""

    DESCRIPTION = (
        "A creativity pass over a prompt. Mark the parts to reimagine inline as "
        "<imagine str=N>...</imagine> — str is that span's imagination strength from 0 to 9 — "
        "and the model rewrites only those spans, splicing each back in wrapped in *asterisks*. "
        "A bare <imagine> uses the node's 'imagination' default. One creative_prompt comes out, "
        "ready to feed into any MiniMax-H3 writer. Runs on any instruction-following GGUF."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": EXAMPLE_PROMPT,
                        "tooltip": (
                            "The prompt to embellish. Wrap the parts to reimagine in "
                            "<imagine str=N>...</imagine>; text outside the tags is kept as is. "
                            "With no tags the prompt is returned unchanged and no model is loaded."
                        ),
                    },
                ),
                "imagination": (
                    "INT",
                    {
                        "default": LEVEL_MAX,
                        "min": LEVEL_MIN,
                        "max": LEVEL_MAX,
                        "step": 1,
                        "tooltip": (
                            "Default imagination strength (0-9) for any <imagine> tag written "
                            "without its own str=. Also substituted for {level} in the system "
                            "prompt. 9 is the most imaginative."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "The whole system message. Use {level} to mark where the default "
                            "imagination strength goes; without it the default is stated in an "
                            "appended line."
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
                "greedy": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Deterministic decoding, which keeps the untouched text intact. Turn "
                            "off (and randomise the seed) for more variation in the reimaginings."
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
                    {"default": False, "tooltip": "Keep the model in VRAM after the rewrite."},
                ),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": BYPASS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("creative_prompt",)
    FUNCTION = "imagine"
    CATEGORY = CATEGORY

    def imagine(
        self,
        prompt,
        imagination,
        system_prompt,
        model,
        greedy,
        seed,
        keep_model_loaded,
        options=None,
        bypass=False,
        unique_id=None,
    ):
        if bypass:
            return _bypassed(unique_id, prompt, ())[:1]

        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        progress = NodeProgress(unique_id)

        if not IMAGINE_RE.search(prompt):
            log.info(
                "[minimax_h3_rewriter.creative_node] no <imagine> spans; returning the prompt "
                "unchanged without loading a model"
            )
            progress.finish("no <imagine> spans — prompt returned unchanged")
            return (prompt.strip(),)

        level = max(LEVEL_MIN, min(LEVEL_MAX, int(imagination)))
        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)

        messages = [
            {"role": "system", "content": _compose_system(system_prompt, level)},
            {"role": "user", "content": prompt},
        ]

        choice = _resolve_writer_choice(model)
        if choice.local:
            model_path = choice.reference
        else:
            model_path = _ensure_file(
                choice.reference,
                choice.file,
                "Writer model",
                settings["auto_download"],
                progress,
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
                f"the prompt needs a {needed}-token context, raising n_ctx from {n_ctx}",
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

        text = (text or "").strip()
        note = _preservation_note(prompt, text)
        progress.text(note + (text[-2000:] if text else "(empty rewrite)"), force=True)
        return (text,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3CreativeImaginer": MiniMaxH3CreativeImaginer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CreativeImaginer": "MiniMax-H3 Creative Imaginer",
}
