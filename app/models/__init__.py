"""Model plugins: CLAP, MERT, OpenL3, plus the registry."""
from __future__ import annotations

from app.models.base import ModelPlugin
from app.models.clap_model import ClapPlugin, CANDIDATE_TAGS
from app.models.mert_model import MertPlugin
from app.models.openl3_model import OpenL3Plugin
from app.models.registry import get_plugin, list_plugins, plugin_info

__all__ = [
    "ModelPlugin",
    "ClapPlugin",
    "CANDIDATE_TAGS",
    "MertPlugin",
    "OpenL3Plugin",
    "get_plugin",
    "list_plugins",
    "plugin_info",
]
