# Copyright 2025 fat-tire and other contributors. Licensed under the GPL, version 3
from .nodes.MediaLoad import MediaLoad
from .nodes.MediaMerge import MediaMerge
from .nodes.MediaSave import MediaSave

NODE_CLASS_MAPPINGS = {
    "MediaSave": MediaSave,
    "MediaLoad": MediaLoad,
    "MediaMerge": MediaMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaSave": "Save Media ⤵️",
    "MediaLoad": "Load Media ⤴️",
    "MediaMerge": "Merge Media 🔀",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']