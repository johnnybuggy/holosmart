"""HoloSmart database layer.

Exposes :class:`~app.db.database.Database` (per-thread connection factory,
idempotent schema bootstrap, transaction helper), the float32 vector
(de)serialisation helpers and the :mod:`~app.db.repo` repository functions.
"""
from __future__ import annotations

from . import repo
from .database import Database, blob_to_vec, vec_to_blob

__all__ = ["Database", "blob_to_vec", "vec_to_blob", "repo"]
