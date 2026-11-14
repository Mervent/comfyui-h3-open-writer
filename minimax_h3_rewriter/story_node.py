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
  The reasoning block the model emits (``<think>...</think>``, or Gemma-4's
  ``<|channel>thought ... <channel|>``) is stripped from the returned story, so
  only the prose comes out.
- **No template.** ``system_prompt`` is a plain writing instruction, the guide
  and the structured fields are gone, and the raw generation is returned as-is.

Connect ``image`` and the same model looks at the pictures before it writes.
That run goes through ``llama-mtmd-cli`` instead (see ``mtmd_engine.write``),
so the model needs its projector (``mmproj``) in the same folder, and it is a
fresh process each time: ``keep_model_loaded`` has no effect there.

Set ``save_dir`` and every story the node returns is also written there as a
plain ``.txt`` file named after a hash of its text, so the folder grows into a
library that a prompt template can pick stories from later.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re

from . import cache, discovery, guide_prompt, media, mtmd_engine
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

# Qwen-style models wrap reasoning in <think>...</think>; Gemma-4 writes it into a
# thought channel, <|channel>thought\n...<channel|>, with the channel name inside.
_THINK_OPEN = r"(?:<think\b[^>]*>|<\|channel>(?:thought\b)?)"
_THINK_CLOSE = r"(?:</think\s*>|<channel\|>)"
THINK_BLOCK_RE = re.compile(rf"(?is){_THINK_OPEN}(?P<body>.*?){_THINK_CLOSE}")
THINK_OPEN_TAG_RE = re.compile(rf"(?is){_THINK_OPEN}")
THINK_CLOSE_RE = re.compile(rf"(?is){_THINK_CLOSE}")

DEFAULT_SYSTEM_PROMPT = """You are a creative fiction writer. Read the user's idea and write one short,
self-contained story based on it. If pictures are attached, build the story on
what they show.

Write only the story itself -- vivid prose with a clear beginning, middle, and
end. Do not add a title, headings, author notes, commentary, or explanations,
and do not restate the idea. Match the language of the user's idea."""

EXAMPLE_PROMPT = "A lighthouse keeper discovers that the light has started answering back."

EFFORT_CHOICES = ["default", "low", "medium", "high", "xhigh"]

#: Pictures taken from one IMAGE batch, spread evenly, as the captioners do.
MAX_PICTURES = media.DEFAULT_MAX_FRAMES

#: Pixels per vision-token side at the finest among common projectors: Qwen2.5-VL
#: merges 14-pixel patches two by two. Newer Qwen merges 16-pixel ones and Gemma
#: spends a fixed budget, so this reserves too much rather than too little -- mtmd
#: fails a run outright when the pictures do not fit the context.
PIXELS_PER_TOKEN_SIDE = 28

#: The marker and the begin/end tokens mtmd wraps around each picture.
TOKENS_AROUND_PICTURE = 16


def _split_think(text: str) -> tuple[str, str]:
    """Separate reasoning from story, returning (story, thinking).

    The split keys off the closing tag, not a matching pair: Qwen-style
    templates pre-inject the opening ``<think>`` into the prompt, so the model's
    output usually carries only the closing tag. Everything before the first
    closing tag (minus a stray leading opener) is reasoning and the rest is the
    story. Fully paired blocks anywhere are pulled out first, and a lone unclosed
    opener is treated as reasoning truncated by ``max_new_tokens``. Both
    ``<think>``/``</think>`` and Gemma-4's ``<|channel>thought``/``<channel|>``
    are recognised. Surrounding newlines are dropped; tag case, attributes, and
    whitespace inside the closing ``</think>`` are ignored.
    """
    text = text or ""
    thoughts = [match.group("body").strip() for match in THINK_BLOCK_RE.finditer(text)]
    story = THINK_BLOCK_RE.sub("", text)
    close = THINK_CLOSE_RE.search(story)
    if close:
        head = THINK_OPEN_TAG_RE.sub("", story[: close.start()], count=1)
        thoughts.append(head.strip())
        story = story[close.end() :]
    else:
        opened = THINK_OPEN_TAG_RE.search(story)
        if opened:
            log.warning(
                "[minimax_h3_rewriter.story_node] a reasoning block was never closed; the story "
                "was probably cut off by max_new_tokens"
            )
            thoughts.append(story[opened.end() :].strip())
            story = story[: opened.start()]
    thinking = "\n\n".join(thought for thought in thoughts if thought).strip()
    return story.strip(), thinking


def _picture_count(image) -> int:
    return min(int(image.shape[0]), MAX_PICTURES)


def _picture_tokens(image) -> int:
    """A generous context reserve for the pictures of an IMAGE batch (batch, height, width, channels)."""
    height, width = int(image.shape[1]), int(image.shape[2])
    per_picture = (
        math.ceil(height / PIXELS_PER_TOKEN_SIDE) * math.ceil(width / PIXELS_PER_TOKEN_SIDE)
        + TOKENS_AROUND_PICTURE
    )
    return _picture_count(image) * per_picture


def _image_digest(image) -> str:
    """A hash of the pixels, so a cached story is never reused for another picture."""
    import numpy

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else numpy.asarray(image)
    digest = hashlib.sha256(repr(array.shape).encode("ascii"))
    digest.update(numpy.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _projector(model_path: str) -> str:
    """The model's own projector, or a refusal that says how to provide one."""
    mmproj_path = discovery.projector_for(model_path)
    if not mmproj_path:
        raise RuntimeError(
            f"'{os.path.basename(model_path)}' has no projector beside it, so it cannot see the "
            f"picture. Put its 'mmproj-*.gguf' into the same folder -- alone with the model, or "
            f"named after it when the folder holds several -- or disconnect 'image'."
        )
    if not discovery.gguf_header(mmproj_path)["vision"]:
        raise RuntimeError(
            f"'{os.path.basename(mmproj_path)}' was built without a vision encoder, so "
            f"'{os.path.basename(model_path)}' cannot see the picture."
        )
    return mmproj_path


def _library_dir(save_dir: str) -> str:
    """The folder to save stories into, created if missing.

    A relative path sits under ComfyUI's output folder."""
    directory = os.path.expanduser(save_dir.strip())
    if not os.path.isabs(directory):
        import folder_paths

        directory = os.path.join(folder_paths.get_output_directory(), directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def _save_story(directory: str, story: str) -> None:
    """Write the story as ``<hash>.txt``; the same story always lands in the same file."""
    name = hashlib.sha256(story.encode("utf-8")).hexdigest()[:16] + ".txt"
    destination = os.path.join(directory, name)
    if os.path.exists(destination):
        return
    part = destination + ".part"
    try:
        with open(part, "w", encoding="utf-8") as handle:
            handle.write(story + "\n")
        os.replace(part, destination)
    except OSError as error:
        log.warning("[minimax_h3_rewriter.story_node] could not save '%s': %s", destination, error)
        if os.path.exists(part):
            os.remove(part)
        return
    log.info("[minimax_h3_rewriter.story_node] saved %s", destination)


class MiniMaxH3StoryWriter:
    """Write a short, free-form story from a prompt, with model reasoning enabled."""

    DESCRIPTION = (
        "A minimal free-form writer: give it an idea and it writes one short story, with no "
        "MiniMax-H3 guide, no structured fields, and no post-parsing. Model reasoning is enabled "
        "and the resulting reasoning block is stripped, so only the prose is returned. Runs on any "
        "instruction-following GGUF through the vendored engine. Connect 'image' and a model "
        "with its mmproj looks at the pictures and writes the story about them."
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
                            "writing; the reasoning is stripped from the output. With 'image' "
                            "connected, the model's 'mmproj-*.gguf' has to sit in the same folder."
                        ),
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Let the model reason before writing (enable_thinking). On a model "
                            "with a thinking mode its reasoning is separated into the 'thinking' "
                            "output; off makes it write directly and leaves 'thinking' empty."
                        ),
                    },
                ),
                "reasoning_effort": (
                    EFFORT_CHOICES,
                    {
                        "default": "default",
                        "tooltip": (
                            "Reasoning depth for models whose chat template reads reasoning_effort "
                            "(e.g. Qwen3.8): low, medium, high, xhigh. 'default' leaves the "
                            "template's own default. Only takes effect with thinking on and a "
                            "supporting model; harmlessly ignored otherwise. Lower effort spends "
                            "fewer tokens on thinking, which helps against truncated stories."
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
                    {
                        "default": False,
                        "tooltip": (
                            "Keep the writer in VRAM after the story. No effect with 'image' "
                            "connected: that run is a separate process, and the model leaves with it."
                        ),
                    },
                ),
                "use_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Reuse a stored story when the prompt, system prompt, pictures, model, "
                            "seed and every setting are unchanged, instead of running the model "
                            "again. Turn off to always regenerate. The 'Open cache folder' button "
                            "shows where the stories are kept."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            f"Pictures for the model to look at before it writes; a batch is up "
                            f"to {MAX_PICTURES} separate pictures, in order. Needs a model with "
                            f"its mmproj in the same folder, e.g. Qwen3.6 or Gemma-4."
                        ),
                    },
                ),
                "options": (OPTIONS_TYPE,),
                "save_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Folder to save every story into as a .txt file, building a library "
                            "to reuse later. A relative path goes under ComfyUI's output folder. "
                            "Files are named after the story's hash, so the same story is never "
                            "saved twice. Empty saves nothing."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("story", "thinking", "raw")
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(
        self,
        prompt,
        system_prompt,
        model,
        thinking,
        reasoning_effort,
        greedy,
        seed,
        keep_model_loaded,
        use_cache=True,
        image=None,
        options=None,
        save_dir="",
        unique_id=None,
    ):
        if not (prompt or "").strip():
            raise ValueError("prompt must not be empty")
        library = _library_dir(save_dir) if (save_dir or "").strip() else ""

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": prompt},
        ]

        effort = "" if reasoning_effort == "default" else reasoning_effort
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
            "enable_thinking": bool(thinking),
            "reasoning_effort": effort,
        }
        if image is not None:
            gen_params["images"] = _image_digest(image)
        cache_key = cache.key(messages, gen_params) if use_cache else ""

        if cache_key:
            cached = cache.load(cache_key)
            if cached is not None:
                story, thinking = _split_think(cached)
                progress.text(
                    "Reused a cached story; the model was not run.\n\n"
                    + (story[-2000:] if story else "(empty story)"),
                    force=True,
                )
                log.info("[minimax_h3_rewriter.story_node] cache hit %s", cache_key)
                if library and story:
                    _save_story(library, story)
                return (story, thinking, cached)

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
        if image is not None:
            needed += _picture_tokens(image)
        if needed > n_ctx:
            progress.text(
                f"the prompt needs a {needed}-token context, raising n_ctx from {n_ctx}",
                force=True,
            )
            n_ctx = needed

        sampling = dict(
            gpu_layers=int(settings["gpu_layers"]),
            n_ctx=n_ctx,
            device=settings["device"],
            progress=progress,
            seed=int(seed),
            greedy=greedy,
            max_new_tokens=max_new_tokens,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            top_k=int(settings["top_k"]),
            repetition_penalty=float(settings["repetition_penalty"]),
            presence_penalty=float(settings["presence_penalty"]),
            enable_thinking=bool(thinking),
            reasoning_effort=effort,
        )
        if image is None:
            text = _gguf_text(
                settings,
                model_path=model_path,
                adapter_path=None,
                keep_loaded=keep_model_loaded,
                messages=messages,
                **sampling,
            )
        else:
            mmproj_path = _projector(model_path)
            if keep_model_loaded:
                log.info(
                    "[minimax_h3_rewriter.story_node] keep_model_loaded has no effect with "
                    "'image' connected: llama-mtmd-cli is a fresh process per run"
                )
            text = mtmd_engine.write(
                model_path=model_path,
                mmproj_path=mmproj_path,
                messages=messages,
                image=image,
                max_images=MAX_PICTURES,
                backend=settings["llama_backend"],
                auto_download=settings["auto_download"],
                **sampling,
            )

        if cache_key:
            cache.store(
                cache_key,
                text=text,
                meta={
                    "model": model,
                    "greedy": bool(greedy),
                    "seed": int(seed),
                    "thinking": bool(thinking),
                    "reasoning_effort": effort,
                    "pictures": _picture_count(image) if image is not None else 0,
                    "prompt_preview": (prompt or "").strip()[:200],
                },
            )

        raw = text or ""
        story, thinking = _split_think(raw)
        if not story and raw.strip():
            progress.text(
                "The model produced only reasoning and no story — it was likely cut off by "
                "max_new_tokens. Raise it and try again.",
                force=True,
            )
        else:
            progress.text(story[-2000:] if story else "(empty story)", force=True)
        if library and story:
            _save_story(library, story)
        return (story, thinking, raw)


NODE_CLASS_MAPPINGS = {"MiniMaxH3StoryWriter": MiniMaxH3StoryWriter}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3StoryWriter": "MiniMax-H3 Story Writer",
}
