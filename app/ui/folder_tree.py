"""Left pane: library tree mirroring the real folder structure, plus a
filter box and Fold all / Unfold all buttons for quick lookups over files
and folders."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QFontMetrics
from PySide6.QtWidgets import (
    QHBoxLayout, QLineEdit, QMenu, QToolButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from app.db import repo
from app.db.database import Database
from app.models.registry import list_plugins

FOLDER_ROLE = Qt.ItemDataRole.UserRole + 1   # db folder id (library roots only)
EXCLUDED_ROLE = Qt.ItemDataRole.UserRole + 4  # truthy: not subject to analysis
TRACK_ROLE = Qt.ItemDataRole.UserRole + 2    # track id (file items)
DIR_ROLE = Qt.ItemDataRole.UserRole + 3      # full directory path (subfolder nodes)

_STATUS_GLYPH = {"new": "•", "analyzing": "…", "analyzed": "✓", "error": "✗"}

#: Column 0 = name, column 1 = overall analysis status, columns 2.. carry one
#: status per registered model plugin (CLAP, MERT, OpenL3, FFT, …).
MODEL_COLUMN_OFFSET = 2

#: Glyph shown in a model column when the track's chunks carry that model's
#: embeddings (:data:`_MODEL_MISSING` marks configured-but-absent results).
_MODEL_DONE = "✓"
_MODEL_MISSING = "✗"
#: Glyph shown in every status column of a track that is not subject to
#: analysis at all (excluded extension — WAV by default).
_MODEL_EXCLUDED = "–"

#: Grey foreground + italic name for tracks excluded from analysis: they
#: neither count toward the analysis-progress percentages nor invite a run.
_EXCLUDED_GREY = QColor("#9e9e9e")

#: Upper bound for the overall Status column.  Its cells hold a glyph or a
#: progress percentage; anything longer lives in the cell tooltip.
_STATUS_COLUMN_MAX_WIDTH = 110

#: Upper bound for the per-model columns (glyphs + progress percentages).
#: Without these caps the columns grow past the tree pane and the status
#: area disappears from view entirely.
_MODEL_COLUMN_MAX_WIDTH = 96

#: Column 0 never shrinks below this, even in a very narrow pane, so a file
#: name stays readable; the leftover pane width (minus the status columns)
#: is the real budget (see :meth:`_autosize_name_column`).
_NAME_MIN_WIDTH = 120


def _status_text_column(index: int) -> int:
    """Item column carrying the status cell for count-entry *index*.

    Index 0 (overall status) lives in column 1; index ``1 + k`` (the *k*-th
    model plugin) lives in column ``MODEL_COLUMN_OFFSET + k``.
    """
    return 1 if index == 0 else MODEL_COLUMN_OFFSET + index - 1

#: Brush painting live activity (scanning library roots; tracks under
#: analysis and their folder chain).  A semi-transparent amber tint instead
#: of solid yellow: vibrant yellow was unreadable with the white item text
#: of dark/night palettes, while a faint gold tint keeps the text readable
#: on both dark and light backgrounds.
_ACTIVITY_BRUSH = QBrush(QColor(255, 200, 0, 60))

#: Dark-green foreground painting the status column of folders whose whole
#: subtree is analysed (dark enough to stay readable on light backgrounds).
_COMPLETE_GREEN = QColor("#1e7e34")


class FolderTree(QTreeWidget):
    """Library root folders as top-level items; below each root the real
    directory hierarchy, with track files nested at their actual location.

    Subfolder nodes carry only ``DIR_ROLE`` (their full path); the owning
    db folder id (``FOLDER_ROLE``) stays on the root item.
    """

    track_selected = Signal(int)   # track_id
    # Emitted when a track file is double-clicked: play it in the system
    # player. Folder nodes keep the default expand/collapse behavior.
    track_play_requested = Signal(int)

    def __init__(self, db: Database, parent=None,
                 excluded_extensions: tuple[str, ...] = ()) -> None:
        super().__init__(parent)
        self._db = db
        self._filter_text = ""
        # Lower-case extensions that are NOT subject to analysis (WAV by
        # default, per AppConfig.analyze_wav): their items are greyed out
        # and they do not count toward the analysis-progress percentages.
        self._excluded_extensions = {e.lower() for e in excluded_extensions}
        # Registered model plugins (process-wide singletons, cheap to list):
        # one status column per plugin, appended after the overall Status.
        self._plugins = list_plugins()
        # Normalized root paths currently in their scan/index phase, kept on
        # the instance so the yellow highlight survives refresh() rebuilds.
        self._scanning_paths: set[str] = set()
        # Normalized paths of tracks currently being analyzed — the track
        # items and their ancestor folder chain paint yellow (see
        # set_analyzing_paths); also survives refresh() rebuilds.
        self._analyzing_paths: set[str] = set()
        self.setHeaderLabels(
            ["Library", "Status"]
            + [plugin.display_name for plugin in self._plugins])
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.itemSelectionChanged.connect(self._on_selection)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Double-click on a track file: play it in the system player."""
        track_id = item.data(0, TRACK_ROLE)
        if track_id is not None:
            self.track_play_requested.emit(int(track_id))

    # ------------------------------------------------------------ population --
    def refresh(self) -> None:
        # Preserve the user's place across the rebuild: current selection
        # (file OR folder), expansion states and scroll offset would all be
        # lost by clear() — e.g. _on_all_analyzed refreshes after a run, and
        # that must not yank focus away from the file the user is on.
        had_items = self.topLevelItemCount() > 0
        current_key = self._current_item_key()
        expanded_keys = self._capture_expanded() if had_items else set()
        scroll = self.verticalScrollBar().value()
        self.blockSignals(True)
        self.clear()
        conn = self._db.connect()
        try:
            # One grouped query serves the per-model status columns of every
            # track: {track_id: {model: chunk-embedding count}}.
            embedding_counts = repo.get_all_track_embedding_counts(conn)
            for folder in repo.list_folders(conn):
                folder_item = QTreeWidgetItem([folder["path"], ""])
                folder_item.setData(0, FOLDER_ROLE, int(folder["id"]))
                folder_item.setToolTip(0, folder["path"])
                self.addTopLevelItem(folder_item)
                for track in repo.list_tracks(conn, int(folder["id"])):
                    self._insert_track(folder_item, folder["path"], track,
                                       embedding_counts)
        finally:
            conn.close()
        self._sort_item(self.invisibleRootItem())
        if expanded_keys:
            self._restore_expanded(expanded_keys)
        else:
            self.expandToDepth(0)
        self.blockSignals(False)
        if self._filter_text:
            self.apply_filter(self._filter_text)
        self._update_folder_statuses()
        self._apply_highlights()
        self._autosize_name_column()
        if had_items:
            # After signals are unblocked so the re-selection propagates to
            # the detail pane exactly like a user click would.  Scroll last,
            # so re-selecting/expanding cannot yank the viewport elsewhere.
            self._restore_current(current_key)
            self.verticalScrollBar().setValue(scroll)

    # ------------------------------------------------- selection preservation --
    @staticmethod
    def _item_key(item: QTreeWidgetItem) -> tuple[str, str]:
        """Stable identity of *item* across refresh() rebuilds."""
        track = item.data(0, TRACK_ROLE)
        if track is not None:
            return ("track", str(int(track)))
        dir_path = item.data(0, DIR_ROLE)
        if dir_path is not None:
            return ("dir", str(dir_path))
        return ("root", item.text(0))

    def _current_item_key(self) -> tuple[str, str] | None:
        item = self.currentItem() or self._selected_item()
        return self._item_key(item) if item is not None else None

    def _capture_expanded(self) -> set[tuple[str, str]]:
        """Keys of every currently expanded folder/root item."""
        keys: set[tuple[str, str]] = set()
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.childCount() and item.isExpanded():
                keys.add(self._item_key(item))
            stack.extend(item.child(i) for i in range(item.childCount()))
        return keys

    def _restore_expanded(self, keys: set[tuple[str, str]]) -> None:
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.childCount() and self._item_key(item) in keys:
                item.setExpanded(True)
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _restore_current(self, key: tuple[str, str] | None) -> None:
        """Re-select the item identified by *key* (no-op when ``None`` or
        gone, e.g. the file was removed from the library)."""
        if key is None:
            return
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if self._item_key(item) == key:
                ancestor = item.parent()
                while ancestor is not None:
                    ancestor.setExpanded(True)
                    ancestor = ancestor.parent()
                self.setCurrentItem(item)
                return
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _insert_track(self, folder_item: QTreeWidgetItem, root_path: str,
                      track, embedding_counts: dict[int, dict[str, int]]) -> None:
        """Add ``track`` under ``folder_item`` nested along its real path."""
        try:
            rel = Path(track["path"]).relative_to(root_path)
        except ValueError:   # not under the root: fall back to a flat child
            rel = Path(track["filename"])
        parent = folder_item
        current = root_path
        for part in rel.parts[:-1]:
            current = str(Path(current) / part)
            parent = self._dir_child(parent, part, current)
        parent.addChild(self._make_track_item(track,
                                              embedding_counts.get(int(track["id"]), {})))

    def _dir_child(self, parent: QTreeWidgetItem, name: str,
                   full_path: str) -> QTreeWidgetItem:
        """Existing child node for ``full_path`` or a new one (no duplicates)."""
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, DIR_ROLE) == full_path:
                return child
        item = QTreeWidgetItem([name, ""])
        item.setData(0, DIR_ROLE, full_path)
        item.setToolTip(0, full_path)
        parent.addChild(item)
        return item

    def _make_track_item(self, track,
                     model_counts: dict[str, int] | None = None) -> QTreeWidgetItem:
        excluded = ((track["extension"] or "").lower()
                    in self._excluded_extensions)
        glyph = (_MODEL_EXCLUDED if excluded
                 else _STATUS_GLYPH.get(track["status"], "•"))
        cells = ([track["filename"], glyph]
                 + [""] * len(self._plugins))
        item = QTreeWidgetItem(cells)
        item.setData(0, TRACK_ROLE, int(track["id"]))
        item.setData(0, EXCLUDED_ROLE, excluded)
        item.setToolTip(0, track["path"])
        if excluded:
            self._apply_excluded_look(item)
        if (os.path.normpath(track["path"]) in self._analyzing_paths):
            # Currently under analysis: yellow like its folder chain (the
            # chain itself is painted by _apply_highlights).
            item.setBackground(0, _ACTIVITY_BRUSH)
        status = track["status"] + (
            f" — {track['status_message']}" if track["status_message"] else "")
        if excluded:
            status += (" — excluded from analysis (WAV files are excluded "
                       "by default; enable analyze_wav in the config)")
        item.setToolTip(1, status)
        self._apply_model_columns(item, track["status"], model_counts or {})
        return item

    def _apply_excluded_look(self, item: QTreeWidgetItem) -> None:
        """Grey out a track item that is not subject to analysis.

        Italic grey filename, grey status glyph, and a grey dash in every
        model column — the item must read as deliberately out of scope, not
        as pending work.
        """
        grey = QBrush(_EXCLUDED_GREY)
        font = item.font(0)
        font.setItalic(True)
        item.setFont(0, font)
        for column in range(2 + len(self._plugins)):
            item.setForeground(column, grey)

    def _apply_model_columns(self, item: QTreeWidgetItem, status: str,
                             model_counts: dict[str, int]) -> None:
        """Write the per-model status glyphs/tooltips of a track item.

        ``model_counts`` maps model name -> number of chunk embeddings stored
        for the track.  A model with results shows :data:`_MODEL_DONE`;
        a model without results on an *analyzed* track shows
        :data:`_MODEL_MISSING` (its plugin was unavailable or failed); an
        unanalyzed track leaves the column empty.
        """
        excluded = bool(item.data(0, EXCLUDED_ROLE))
        for index, plugin in enumerate(self._plugins):
            column = MODEL_COLUMN_OFFSET + index
            count = int(model_counts.get(plugin.name, 0))
            if excluded:
                item.setText(column, _MODEL_EXCLUDED)
                tip = (f"{plugin.display_name}: excluded from analysis "
                       "(WAV files are excluded by default)")
                if count > 0:
                    tip += f" — {count} chunk embeddings stored"
                item.setToolTip(column, tip)
                continue
            if count > 0:
                item.setText(column, _MODEL_DONE)
                item.setToolTip(
                    column,
                    f"{plugin.display_name}: {count} chunk embeddings")
            elif status == "analyzed":
                item.setText(column, _MODEL_MISSING)
                item.setToolTip(column,
                                f"{plugin.display_name}: no results")
            else:
                item.setText(column, "")
                item.setToolTip(column, "")

    def _sort_item(self, item: QTreeWidgetItem) -> None:
        """Deterministic order: directories first, then files (case-insensitive)."""
        children = item.takeChildren()
        children.sort(key=self._sort_key)
        for child in children:
            item.addChild(child)
            self._sort_item(child)

    @staticmethod
    def _sort_key(item: QTreeWidgetItem) -> tuple:
        is_dir = (item.data(0, DIR_ROLE) is not None
                  or item.data(0, FOLDER_ROLE) is not None)
        name = item.text(0)
        return (0 if is_dir else 1, name.casefold(), name)

    # ------------------------------------------------------------- selection --
    def _on_selection(self) -> None:
        track_id = self.selected_track_id()
        if track_id is not None:
            self.track_selected.emit(track_id)

    def _selected_item(self) -> QTreeWidgetItem | None:
        items = self.selectedItems()
        return items[0] if items else None

    def selected_track_id(self) -> int | None:
        item = self._selected_item()
        if item is None:
            return None
        return item.data(0, TRACK_ROLE)

    def selected_folder_id(self) -> int | None:
        """Owning library root id: walk up from the selected item (file or
        subfolder node) to the ancestor carrying ``FOLDER_ROLE``."""
        item = self._selected_item()
        while item is not None:
            data = item.data(0, FOLDER_ROLE)
            if data is not None:
                return int(data)
            item = item.parent()
        return None

    def selected_track_ids_in_folder(self) -> list[int]:
        """Track ids of the selected file, or of every file in the selected
        (sub)folder's subtree."""
        item = self._selected_item()
        if item is None:
            return []
        track_id = item.data(0, TRACK_ROLE)
        if track_id is not None:
            return [int(track_id)]
        return self._collect_track_ids(item)

    def folder_track_ids_for_references(self) -> list[int]:
        """All track ids of the selected folder subtree.

        Like :meth:`selected_track_ids_in_folder`, but when a FILE is
        selected the PARENT folder's whole subtree is used — this backs
        "add a full folder as multiple references".
        """
        item = self._selected_item()
        if item is None:
            return []
        if item.data(0, TRACK_ROLE) is not None:
            item = item.parent()
        if item is None:
            return []
        return self._collect_track_ids(item)

    @staticmethod
    def _collect_track_ids(item: QTreeWidgetItem) -> list[int]:
        ids: list[int] = []

        def walk(node: QTreeWidgetItem) -> None:
            for i in range(node.childCount()):
                child = node.child(i)
                track_id = child.data(0, TRACK_ROLE)
                if track_id is not None:
                    ids.append(int(track_id))
                walk(child)

        walk(item)
        return ids

    def select_track(self, track_id: int) -> None:
        """Programmatically select the item of ``track_id`` (after refresh),
        expanding its ancestors as needed."""
        item = self._find_track_item(self.invisibleRootItem(), int(track_id))
        if item is None:
            return
        ancestor = item.parent()
        while ancestor is not None:
            ancestor.setExpanded(True)
            ancestor = ancestor.parent()
        self.setCurrentItem(item)

    @staticmethod
    def _find_track_item(item: QTreeWidgetItem,
                         track_id: int) -> QTreeWidgetItem | None:
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, TRACK_ROLE) == track_id:
                return child
            found = FolderTree._find_track_item(child, track_id)
            if found is not None:
                return found
        return None

    # --------------------------------------------------- in-place status UX --
    def update_track_status(self, track_id: int, status: str,
                            status_message: str | None = None) -> None:
        """Update one track's status in place, without a rebuild.

        Unlike :meth:`refresh` — which rebuilds the whole tree and thereby
        resets the user's selection, expansion states and scroll position —
        this rewrites the track item's column-1 glyph and tooltip (same
        format as a fresh build: ``status``, plus ``" — message"`` when a
        message is given; column-0 tooltip keeps the path), refreshes the
        per-model status columns (2..) from the stored chunk embeddings and
        recomputes the analysis-progress percentages of its ancestor
        folder/root nodes in every status column.  A track that is not in
        the tree is a no-op.
        """
        item = self._find_track_item(self.invisibleRootItem(), int(track_id))
        if item is None:
            return
        excluded = bool(item.data(0, EXCLUDED_ROLE))
        item.setText(1, _MODEL_EXCLUDED if excluded
                     else _STATUS_GLYPH.get(status, "•"))
        tip = status + (f" — {status_message}" if status_message else "")
        if item.data(0, EXCLUDED_ROLE):
            tip += (" — excluded from analysis (WAV files are excluded "
                    "by default; enable analyze_wav in the config)")
        item.setToolTip(1, tip)
        if item.data(0, EXCLUDED_ROLE):
            self._apply_excluded_look(item)
        conn = self._db.connect()
        try:
            counts = repo.get_track_embedding_counts(conn, int(track_id))
        finally:
            conn.close()
        self._apply_model_columns(item, status, counts)
        self._update_ancestor_statuses(item)

    def _update_folder_statuses(self) -> None:
        """Recompute every folder/root node's ``"{pct}%"`` analysis-progress
        text and tooltip across all status columns (full walk, used after
        :meth:`refresh`)."""
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            counts = self._subtree_status_counts(child)
            self._write_folder_statuses(child, counts)

    def _update_ancestor_statuses(self, item: QTreeWidgetItem) -> None:
        """Recompute only the folder/root ancestor chain of ``item`` — the
        cheap counterpart of :meth:`_update_folder_statuses` used by
        :meth:`update_track_status` (deeper counts are already current)."""
        node = item.parent()
        while node is not None:
            counts = self._subtree_status_counts(node)
            self._write_folder_statuses(node, counts)
            node = node.parent()

    def _subtree_status_counts(
        self, item: QTreeWidgetItem
    ) -> list[tuple[int, int]]:
        """Count ``(ready, total)`` per status column in *item*'s subtree.

        One entry per status column — index 0 for the overall column 1, then
        one per registered model plugin (columns ``MODEL_COLUMN_OFFSET..``).
        Folder statuses nested inside the subtree are recomputed and written
        in the same bottom-up pass.  A file counts as ready in a column when
        the glyph shown in that column says so (``✓``) — the tree is the
        source of truth for display, independent of the database.
        """
        ready = [[0, 0] for _ in range(1 + len(self._plugins))]
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, EXCLUDED_ROLE):
                # Not subject to analysis: never counts toward progress.
                continue
            if child.data(0, TRACK_ROLE) is not None:
                for index in range(len(ready)):
                    ready[index][1] += 1
                    if child.text(_status_text_column(index)) == _MODEL_DONE:
                        ready[index][0] += 1
                continue
            sub_counts = self._subtree_status_counts(child)
            self._write_folder_statuses(child, sub_counts)
            for index, (sub_ready, sub_total) in enumerate(sub_counts):
                ready[index][0] += sub_ready
                ready[index][1] += sub_total
        return [(r, t) for r, t in ready]

    def _write_folder_statuses(self, item: QTreeWidgetItem,
                               counts: list[tuple[int, int]]) -> None:
        """Write *item*'s folder status columns/tooltips from per-column
        ``counts`` (overall first, then one entry per model plugin).

        Each column shows the analysis progress as a bare percentage
        (``"{pct}%"``) — the exact ``ready/total`` counts live in the cell
        tooltip; an empty subtree stays empty.  A fully analysed column
        (``ready == total > 0``) additionally gets the dark-green
        :data:`_COMPLETE_GREEN` foreground; any other state clears it again,
        so folders revert when a new file appears or a file flips back to
        non-analysed.  Only the status columns are coloured — the folder
        name in column 0 keeps its plain look (and the yellow scan
        background) untouched.
        """
        # Column 1 (overall) first.
        ready, total = counts[0]
        if total <= 0:
            item.setText(1, "")
            item.setToolTip(1, "")
            item.setForeground(1, QBrush())
        else:
            item.setText(1, f"{ready / total * 100:.0f}%")
            item.setToolTip(1, f"{ready} of {total} files analyzed")
            item.setForeground(
                1, QBrush(_COMPLETE_GREEN) if ready == total else QBrush())
        # One percentage cell per model plugin column.
        for index, plugin in enumerate(self._plugins):
            if index + 1 >= len(counts):   # defensive: shorter counts list
                break
            column = MODEL_COLUMN_OFFSET + index
            m_ready, m_total = counts[index + 1]
            if m_total <= 0:
                item.setText(column, "")
                item.setToolTip(column, "")
                item.setForeground(column, QBrush())
                continue
            item.setText(column,
                         f"{m_ready / m_total * 100:.0f}%")
            item.setToolTip(
                column,
                f"{m_ready} of {m_total} files analyzed with "
                f"{plugin.display_name}")
            item.setForeground(
                column,
                QBrush(_COMPLETE_GREEN) if m_ready == m_total else QBrush())

    # ------------------------------------------------- filename column width --
    def _autosize_name_column(self) -> None:
        """Widen column 0 so nested filenames and their paths fit, capped so
        every status column stays visible.

        The required width accounts for the tree's visual nesting: every
        directory level adds one :meth:`indentation` step (that is the visible
        "path" to the file), on top of the widest column-0 text (long file
        names, and full paths on the root-folder items), plus a fixed
        allowance for the branch control and text margins.

        That width is then capped to the viewport space left over after the
        Status + per-model columns take theirs — a huge library path must
        never push the analysis-status columns out of the visible pane (they
        are the point of the columns). Long names stay reachable: the full
        path rides on every item's tooltip and the user can resize column 0
        or the splitter freely; :meth:`resizeEvent` re-applies the cap when
        the pane changes size.  Called after every :meth:`refresh`.
        """
        metrics = QFontMetrics(self.font())
        indent = self.indentation()
        needed = 0

        def walk(node: QTreeWidgetItem, depth: int) -> None:
            nonlocal needed
            required = (metrics.horizontalAdvance(node.text(0))
                        + depth * indent + 40)
            if required > needed:
                needed = required
            for i in range(node.childCount()):
                walk(node.child(i), depth + 1)

        walk(self.invisibleRootItem(), 0)

        # Size the status columns first (clamped: a "12/34 (35%)" cell or a
        # short header label must not eat the whole pane), then give column 0
        # whatever viewport width is left over.
        reserved = 0
        for column in range(1, self.columnCount()):
            self.resizeColumnToContents(column)
            limit = (_STATUS_COLUMN_MAX_WIDTH if column == 1
                     else _MODEL_COLUMN_MAX_WIDTH)
            width = min(self.header().sectionSize(column), limit)
            self.header().resizeSection(column, width)
            reserved += width + 1

        # Budget from the widget's own width: viewport() can lag behind on
        # not-yet-shown widgets, and resizeEvent must rebalance immediately
        # when the pane changes size.  Subtract the frame and the vertical
        # scrollbar when it is actually visible.
        scrollbar = (self.verticalScrollBar().width()
                     if self.verticalScrollBar().isVisible() else 0)
        budget = max(self.width() - 2 * self.frameWidth() - scrollbar, 200)
        cap = max(budget - reserved, _NAME_MIN_WIDTH)
        width = max(min(needed, cap), 120)
        self.header().resizeSection(0, width)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Re-apply the name-column cap when the pane is resized (window
        resizes, splitter drags) so the status columns can never drift out
        of the visible area."""
        super().resizeEvent(event)
        self._autosize_name_column()

    # ------------------------------------------------------- scan highlight --
    def set_scanning_paths(self, paths: list[str]) -> None:
        """Mark the root folders currently being scanned with a yellow
        background; ``[]`` clears the highlight.

        Only library-root items are marked (phase-2 indexing runs per root
        folder, not per subfolder).  The normalized paths are stored on the
        instance and re-applied by :meth:`refresh`, so the highlight
        survives tree rebuilds.
        """
        self._scanning_paths = {
            os.path.abspath(os.path.expanduser(os.fspath(p)))
            for p in (paths or [])
        }
        self._apply_highlights()

    def set_analyzing_paths(self, paths: list[str]) -> None:
        """Highlight the tracks currently being analyzed, in yellow.

        Every *file* item whose path is in *paths* gets a yellow column-0
        background, and so does every folder/root item above it, recursively
        up to the topmost library root — so an in-flight batch is visible at
        a glance even when its folders are collapsed.  ``[]`` clears the
        highlight.  The normalized paths are stored on the instance and
        re-applied by :meth:`refresh`, so the highlight survives rebuilds.
        """
        self._analyzing_paths = {
            os.path.normpath(os.fspath(p)) for p in (paths or [])
        }
        self._apply_highlights()

    def _apply_highlights(self) -> None:
        """Paint (or clear) the yellow column-0 backgrounds for both live
        activity highlights: scanning library roots and analyzing tracks
        with their whole ancestor folder chain.  One pass so the two
        highlights never erase each other."""
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            scanning = (os.path.abspath(os.path.expanduser(child.text(0)))
                        in self._scanning_paths)
            self._paint_activity(child, scanning)

    def _paint_activity(self, item: QTreeWidgetItem,
                        scanning_root: bool) -> bool:
        """Apply highlights to *item* and its subtree; returns whether the
        subtree contains an analyzing track (the parent uses this to paint
        its own folder chain, recursively up to the root)."""
        subtree_has = False
        for i in range(item.childCount()):
            if self._paint_activity(item.child(i), False):
                subtree_has = True
        if item.data(0, TRACK_ROLE) is not None:
            subtree_has = (os.path.normpath(str(item.toolTip(0)))
                           in self._analyzing_paths)
        if subtree_has or scanning_root:
            item.setBackground(0, _ACTIVITY_BRUSH)
        else:
            item.setBackground(0, QBrush())
        return subtree_has

    # ---------------------------------------------------------------- filter --
    def apply_filter(self, text: str) -> None:
        """Show only the items relevant to ``text`` (case-insensitive
        substring match against column-0 names of files and directories).

        An item stays visible when its own name matches, when one of its
        descendants matches, or when a matching ancestor forces its whole
        subtree visible — so a matching directory keeps all files under it
        reachable and is expanded, as are ancestors of matching items.
        Everything else is hidden.  An empty query restores the unfiltered
        view (root level expanded, deeper levels collapsed, as in refresh).
        """
        self._filter_text = text
        needle = text.strip().casefold()
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            self._filter_item(root.child(i), needle, False, 0)

    def _filter_item(self, item: QTreeWidgetItem, needle: str,
                     force_visible: bool, depth: int) -> bool:
        """Filter ``item`` and its subtree; return True if it stays visible."""
        name_match = (needle in item.text(0).casefold()) if needle else True
        child_match = False
        for i in range(item.childCount()):
            if self._filter_item(item.child(i), needle,
                                 force_visible or name_match, depth + 1):
                child_match = True
        visible = force_visible or name_match or child_match
        item.setHidden(not visible)
        if not needle:
            item.setExpanded(depth == 0)
        elif visible and (child_match or (name_match and depth > 0)):
            item.setExpanded(True)
        return visible

    # ----------------------------------------------------------- context menu --
    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        track_id = self.selected_track_id()
        folder_id = self.selected_folder_id()
        analyze = QAction("Analyze", self)
        find_similar = QAction("Find similar…", self)
        clear = QAction("Clear analysis results", self)
        reveal = QAction("Reveal in Finder", self)
        menu.addAction(analyze)
        menu.addAction(find_similar)
        menu.addSeparator()
        menu.addAction(clear)
        menu.addSeparator()
        menu.addAction(reveal)
        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        window = self.window()
        if chosen is analyze and folder_id is not None:
            ids = self.selected_track_ids_in_folder()
            if ids and hasattr(window, "analyze_track_ids"):
                # A folder selection analyzes the subtree RECURSIVELY and
                # incrementally: already-analyzed tracks are visited too,
                # but analyze_track keeps their chunks and only fills in
                # models that do not cover every chunk yet (so enabling a
                # new model and analyzing the folder adds it everywhere
                # without redoing or dropping the other models' work).
                window.analyze_track_ids(ids, skip_analyzed=False)
        elif chosen is find_similar and track_id is not None:
            if hasattr(window, "open_similar_for"):
                window.open_similar_for(track_id)
        elif chosen is clear and folder_id is not None:
            ids = self.selected_track_ids_in_folder()
            if ids and hasattr(window, "clear_analysis_results"):
                window.clear_analysis_results()
        elif chosen is reveal:
            target = self._reveal_target()
            if target is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _reveal_target(self) -> Path | None:
        """Directory to open: the file's parent, or the folder node itself."""
        item = self._selected_item()
        if item is None:
            return None
        if item.data(0, TRACK_ROLE) is not None:
            return Path(item.toolTip(0)).parent
        path = item.data(0, DIR_ROLE)
        if path is None:   # library root folder: text(0) is the full path
            return Path(item.text(0))
        return Path(str(path))


class LibraryPane(QWidget):
    """Left pane: filter box and Fold all / Unfold all buttons above the
    folder tree.

    Typing filters the tree live (debounced); :meth:`apply_filter` applies
    a query immediately.  :meth:`fold_all` / :meth:`unfold_all` collapse /
    expand every folder node.  Interplay with the other controls:
    ``refresh()`` still resets the expansion to root-level-only and
    re-applies the active filter, and a non-empty filter query auto-expands
    matching branches, which overrides manual folding.
    """

    def __init__(self, db: Database, parent=None,
                 excluded_extensions: tuple[str, ...] = ()) -> None:
        super().__init__(parent)
        self.tree = FolderTree(db, self,
                               excluded_extensions=excluded_extensions)
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("Filter files and folders…")
        self._filter_edit.setClearButtonEnabled(True)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(
            lambda: self.tree.apply_filter(self._filter_edit.text()))
        self._filter_edit.textChanged.connect(
            lambda _text: self._debounce.start())
        self._fold_button = self._tool_button(
            "Fold all", "Collapse all folders", self.fold_all)
        self._unfold_button = self._tool_button(
            "Unfold all", "Expand all folders", self.unfold_all)
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self._filter_edit, 1)
        row_layout.addWidget(self._fold_button)
        row_layout.addWidget(self._unfold_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(row)
        layout.addWidget(self.tree)

    @staticmethod
    def _tool_button(text: str, tooltip: str, slot) -> QToolButton:
        """Compact flat button that never takes focus from the tree."""
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(slot)
        return button

    def fold_all(self) -> None:
        """Collapse every folder node in the tree (root folders included).

        A no-op on an empty tree.  Note that ``refresh()`` afterwards
        resets the expansion to root-level-only, and a non-empty filter
        query re-expands matching branches.
        """
        self.tree.collapseAll()

    def unfold_all(self) -> None:
        """Expand every folder node in the tree (root folders included).

        A no-op on an empty tree.  ``refresh()`` still resets the
        expansion to root-level-only and re-applies the active filter, and
        a non-empty filter query auto-expands matching branches, which
        overrides manual folding.
        """
        self.tree.expandAll()

    def apply_filter(self, text: str) -> None:
        """Apply ``text`` immediately (no debounce) and sync the filter box."""
        self._debounce.stop()
        self._filter_edit.blockSignals(True)
        self._filter_edit.setText(text)
        self._filter_edit.blockSignals(False)
        self.tree.apply_filter(text)
