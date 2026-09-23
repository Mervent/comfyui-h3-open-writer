"""A free-form short-story writer: no H3 schema, reasoning left on.

This is the smallest useful writer in the pack. Unlike the guided/Ref2VA
writers, it does not fetch a MiniMax-H3 guide, does not impose the six-field
output contract, and does not parse the result into sections -- it takes a
prompt, runs any instruction-following GGUF, and returns whatever short story
the model wrote.

Two deliberate differences from the other writers:

- **Reasoning is enabled.** Every other node forces ``enable_thinking=False``
  through the vendored engine; this one passes ``enable_thinking=True`` so a
  model that supports a thinking phase (e.g. Qwen3) may reason before writing.
  The ``<think>...</think>`` block the model emits is stripped from the returned
  story, so only the prose comes out.
- **No template.** ``system_prompt`` is a plain writing instruction, the guide
  and the structured fields are gone, and the raw generation is returned as-is.
"""

from __future__ import annotations

import logging

from . import discovery, guide_prompt
from .nodes import (
    CATEGORY,
    DEFAULT_OPTIONS,
    OPTIONS_TYPE,
    _ensure_file,
    _gguf_text,
    _resolve_writer_choice,
    writer_choices,
)
from .progress import NodeProgress

log = logging.getLogger(__name__)

THINK_CLOSE = "</think>"

DEFAULT_SYSTEM_PROMPT = """You are a creative fiction writer. Read the user's idea and write one short,
self-contained story based on it.

Write only the story itself -- vivid prose with a clear beginning, middle, and
end. Do not add a title, headings, author notes, commentary, or explanations,
and do not restate the idea. Match the language of the user's idea."""

EXAMPLE_PROMPT = "A lighthouse keeper discovers that the light has started answering back."


def _strip_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` reasoning block, keep the story.

    A model told to reason emits its thinking before the prose, closed by
    ``</think>``. Everything up to and including the first such marker is
    reasoning, not story, so it is cut; text without the marker is returned
    unchanged. This handles both a model that writes the opening ``<think>``
    tag itself and a chat template that pre-injects it.
    """
    marker = text.find(THINK_CLOSE)
    if marker != -1:
        text = text[marker + len(THINK_CLOSE) :]
    return text.strip()


class MiniMaxH3StoryWriter:
    """Write a short, free-form story from a prompt, with model reasoning enabled."""

    DESCRIPTION = (
        "A minimal free-form writer: give it an idea and it writes one short story, with no "
        "MiniMax-H3 guide, no structured fields, and no post-parsing. Model reasoning is enabled "
        "and the resulting <think> block is stripped, so only the prose is returned. Runs on any "
        "instruction-following GGUF through the vendored engine."
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
                        "tooltip": "The idea to turn into a short story.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "The whole system message -- a plain writing instruction. No {guide} "
                            "or field contract here; whatever the model writes is returned."
                        ),
                    },
                ),
                "model": (
                    writer_choices(),
                    {
                        "tooltip": (
                            "Any GGUF language model. Entries prefixed 'on disk:' are already in "
                            "your ComfyUI model folders; the rest are fetched on first use. No LoRA "
                            "is applied. A model with a thinking mode (e.g. Qwen3) reasons before "
                            "writing; the reasoning is stripped from the output."
                        ),
                    },
                ),
                "greedy": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Deterministic decoding. Off (with a random seed) gives livelier stories.",
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
                    {"default": False, "tooltip": "Keep the writer in VRAM after the story."},
                ),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story",)
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        prompt,
        system_prompt,
        model,
        greedy,
        seed,
        keep_model_loaded,
        options=None,
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": prompt},
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
            enable_thinking=True,
        )

        story = _strip_think(text or "")
        progress.text(story[-2000:] if story else "(empty story)", force=True)
        return (story,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3StoryWriter": MiniMaxH3StoryWriter}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3StoryWriter": "MiniMax-H3 Story Writer",
}
