"""A creativity pass over a prompt, driven by inline tags.

The user marks spans in the prompt and the model rewrites the whole prompt in a
single pass, changing only those spans and copying everything else verbatim. Two
kinds of span are understood::

    <imagine str=3>a quiet street</imagine>        -> a variation on that text
    <write str=7>describe what she does next</write> -> text written to that brief

``str`` is a 0-9 level: for ``<imagine>`` it is how far the variation may stray
from the original, for ``<write>`` how freely the instruction is interpreted; a
bare tag uses the node's ``imagination`` default. One ``creative_prompt`` comes
out, ready to feed into any of the MiniMax-H3 writer nodes.
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

TAG_RE = re.compile(
    r"<(?P<kind>imagine|write)(?:\s+str=(?P<str>\d+))?\s*>(?P<body>.*?)</(?P=kind)>",
    re.DOTALL | re.IGNORECASE,
)

LEVEL_MIN = 0
LEVEL_MAX = 9

DEFAULT_SYSTEM_PROMPT = """You are given a prompt that contains marked spans. Return the ENTIRE prompt
back, changing only the marked spans and copying every character outside them
exactly as it is -- verbatim, in the same order. Do not answer or react to the
prompt; only rewrite it.

There are two kinds of span:

  <imagine str=N>TEXT</imagine>
      Replace the whole span with a variation on TEXT -- a fresh take, not a
      decorated copy: change the phrasing, angle, or intent rather than adding
      adjectives. N (0-9) is how far the variation may stray from TEXT.

  <write str=N>INSTRUCTION</write>
      INSTRUCTION describes what to write at this spot. Replace the whole span
      with text that carries it out. If it offers options to choose from, pick
      one and write it out; never list the options or say that you chose. N
      (0-9) is how freely you interpret the instruction.

A span written without str= uses the default level of {level}.

Rules:
- Return the complete prompt. Copy everything outside the spans exactly: do not
  add, drop, reorder, or reword any of it.
- Replace each span inline with your result and remove the <imagine>/<write>
  tags. Do not wrap a result in quotation marks, asterisks, or labels.
- Wrap any direct speech in your result inside <d> and </d>, like
  <d>some words</d> -- the exact words a character says aloud, never narration.
- Fit each replacement to the words around it so the sentence still reads.
- Keep the language of the surrounding text -- and, for <write>, of the
  instruction -- unless told otherwise.
- Output only the rewritten prompt: no commentary, no headings, no notes.

Level scale (str=N), for both tags:
  0    minimal: <imagine> barely strays; <write> is plain and literal
  1-2  a light touch, close to the source
  3-4  a clear, vivid change, still recognisably related
  5-6  inventive: reinterpret freely, the mood or angle may shift
  7-8  bold: a surprising take whose meaning departs sharply
  9    wild: the most unexpected reading, even the opposite
       (e.g. <imagine> "Hello, my friend" -> "Prepare to die, bastard.")"""

EXAMPLE_PROMPT = (
    "A woman walks down <imagine str=3>a quiet street</imagine> at night. "
    "<write str=7>Describe what she does when she stops, in one sentence.</write>"
)

LEVEL_LINE = "\n\nSpans written without str= use the default level of {level}."


def _compose_system(system_prompt: str, level: int) -> str:
    """Fill the default level into the system prompt the user supplied.

    ``{level}`` is where the default belongs, so it is substituted there when
    present. A system prompt that never mentions it -- one the user rewrote from
    scratch -- gets the default stated in one appended line rather than silently,
    so a bare tag still has a defined level.
    """
    system_prompt = (system_prompt or "").strip()
    if "{level}" in system_prompt:
        return system_prompt.replace("{level}", str(level))
    return system_prompt + LEVEL_LINE.replace("{level}", str(level))


def _outside_segments(prompt: str) -> list[str]:
    """The literal text between the tags -- everything that must survive verbatim."""
    segments: list[str] = []
    last = 0
    for match in TAG_RE.finditer(prompt):
        segments.append(prompt[last : match.start()])
        last = match.end()
    segments.append(prompt[last:])
    return segments


def _preservation_note(prompt: str, output: str) -> str:
    """Warn -- without failing -- when the model dropped the untouched text.

    A whole-prompt rewrite trusts the model to copy everything outside the spans
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
        "rewrite; the model may have altered text outside the marked spans",
        len(missing),
        len(anchors),
    )
    return (
        f"⚠ {len(missing)} of {len(anchors)} untouched segment(s) are not in the output — "
        f"the model may have changed text outside the marked spans. Lower the temperature "
        f"or try a larger model.\n\n"
    )


class MiniMaxH3CreativeImaginer:
    """Rewrite the ``<imagine>``/``<write>`` spans of a prompt, copying the rest verbatim."""

    DESCRIPTION = (
        "A creativity pass over a prompt. Mark spans inline as <imagine str=N>...</imagine> "
        "(a variation on that text) or <write str=N>...</write> (text written to that "
        "instruction); str is a 0-9 level. The model returns the whole prompt with only those "
        "spans changed and everything else kept verbatim. A bare tag uses the node's "
        "'imagination' default. One creative_prompt comes out, ready for any MiniMax-H3 writer."
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
                            "The prompt to rework. Mark spans as <imagine str=N>...</imagine> "
                            "(vary that text) or <write str=N>...</write> (write to that "
                            "instruction); everything else is kept verbatim. With no tags the "
                            "prompt is returned unchanged and no model is loaded."
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
                            "Default level (0-9) for any <imagine> or <write> tag written without "
                            "its own str=. Also substituted for {level} in the system prompt. For "
                            "<imagine>, 9 strays furthest; for <write>, 9 is the freest reading."
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
                            "level goes; without it the default is stated in an appended line."
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
                            "off (and randomise the seed) for more surprise in the changes."
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

        if not TAG_RE.search(prompt):
            log.info(
                "[minimax_h3_rewriter.creative_node] no <imagine>/<write> tags; returning the "
                "prompt unchanged without loading a model"
            )
            progress.finish("no tags — prompt returned unchanged")
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
