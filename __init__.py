"""ComfyUI entry point for the open MiniMax-H3 writer (vendored engine)."""

import logging

from .minimax_h3_rewriter.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .minimax_h3_rewriter import open_node, writer_8b

log = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS.update(open_node.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(open_node.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(writer_8b.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(writer_8b.NODE_DISPLAY_NAME_MAPPINGS)

try:
    from .minimax_h3_rewriter import multi_caption

    NODE_CLASS_MAPPINGS.update(multi_caption.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(multi_caption.NODE_DISPLAY_NAME_MAPPINGS)
except Exception:
    log.warning(
        "[h3-open-writer] 'Multi Reference Caption' needs a newer ComfyUI than this one, "
        "so it is not registered. Every other node is unaffected.",
        exc_info=True,
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
