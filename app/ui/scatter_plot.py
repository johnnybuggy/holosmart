"""A lightweight scatter-plot canvas for the Visualisation dialog.

The project carries no matplotlib dependency, so the plot is drawn with a
plain ``QPainter``: auto-scaled axes, ~5 grid lines/tick labels per axis,
thousands of points via ``drawPoints`` and a nearest-point hover tooltip.
Colours come from the widget palette, so night/dark mode stays readable.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

#: Plot margins (pixels): room for the rotated Y label, tick labels, X label.
_MARGIN_LEFT = 74
_MARGIN_TOP = 14
_MARGIN_RIGHT = 18
_MARGIN_BOTTOM = 48

#: Pick radius (pixels) for the hover tooltip.
_PICK_RADIUS = 10.0

#: Distinct point colours, cycled per track.
_POINT_COLORS = (
    "#4f8ef7", "#f77f4f", "#4fc07a", "#b24ff7", "#f7c94f",
    "#4fd7f7", "#f74f8e", "#8ef74f", "#a086f7", "#f79f4f",
)

#: Cap on distinct colour groups actually drawn; beyond this the palette
#: cycles meaninglessly and everything is drawn in one accent colour instead.
_MAX_COLOR_GROUPS = 12


def _format_tick(value: float) -> str:
    """Compact tick label: ``%.4g`` trims both 0.00123 and 123456."""
    if value == 0.0:
        return "0"
    return f"{value:.4g}"


class ScatterCanvas(QWidget):
    """Plain-painter 2-D scatter with hover tooltips, click picking and
    axis labels."""

    #: The user clicked a dot (single click, within :data:`_PICK_RADIUS`);
    #: the argument is the point's index in the plotted arrays.
    point_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._xs = np.zeros(0)
        self._ys = np.zeros(0)
        self._labels: list[str] = []
        self._colors: list[str] = []          # per-point colour index
        self._x_label = ""
        self._y_label = ""
        self._status = ""                     # "No data yet" style note
        self._px: np.ndarray | None = None    # cached pixel coords (n, 2)
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._polys: dict[int, QPolygonF] | None = None
        self._poly_size: tuple[int, int] | None = None

    # ---- public API ---------------------------------------------------------
    def set_plot(self, xs, ys, x_label: str = "", y_label: str = "",
                 labels: list[str] | None = None,
                 colors: list[int] | None = None,
                 status: str = "") -> None:
        """Replace the plotted data.

        *xs*/*ys* are equal-length 1-D arrays; *labels* (optional) supply the
        hover tooltip text per point; *colors* (optional) index the palette
        per point (e.g. one index per track); *status* is an optional note
        drawn bottom-left ("subsampled 5 000 of 61 000 points").
        """
        xs_arr = np.asarray(xs, dtype=np.float64).reshape(-1)
        ys_arr = np.asarray(ys, dtype=np.float64).reshape(-1)
        if xs_arr.size != ys_arr.size:
            raise ValueError("x and y must have the same length")
        self._xs, self._ys = xs_arr, ys_arr
        self._labels = list(labels) if labels is not None else []
        self._colors = list(colors) if colors is not None else []
        self._x_label, self._y_label = x_label, y_label
        self._status = status
        self._px = None
        # Derived caches MUST be dropped: polygons were built from the
        # previous point cloud, and while the widget size stays the same
        # paintEvent would happily re-draw the stale plot.
        self._polys = None
        self._poly_size = None
        self.update()

    def clear_plot(self, status: str = "") -> None:
        """Drop all points (optionally leaving a note on the empty plot)."""
        self.set_plot([], [], status=status)

    @property
    def point_count(self) -> int:
        return int(self._xs.size)

    # ---- coordinate mapping -------------------------------------------------
    def _plot_rect(self):
        return self.contentsRect().adjusted(
            _MARGIN_LEFT, _MARGIN_TOP, -_MARGIN_RIGHT, -_MARGIN_BOTTOM)

    @staticmethod
    def _padded_range(values: np.ndarray) -> tuple[float, float]:
        """Data range with 5% headroom; a flat axis gets a ±0.5 window."""
        if values.size == 0:
            return 0.0, 1.0
        lo, hi = float(values.min()), float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0.0, 1.0
        if hi - lo < 1e-12:
            pad = 0.5 if lo == hi else max(abs(lo) * 0.05, 1e-6)
            return lo - pad, hi + pad
        pad = (hi - lo) * 0.05
        return lo - pad, hi + pad

    def _value_to_pixel(self, value: float, lo: float, hi: float,
                        pix_lo: int, pix_hi: int) -> float:
        if hi <= lo:
            return float((pix_lo + pix_hi) / 2)
        frac = (value - lo) / (hi - lo)
        return pix_lo + frac * (pix_hi - pix_lo)

    def _pixel_to_value(self, pixel: float, lo: float, hi: float,
                        pix_lo: int, pix_hi: int) -> float:
        if pix_hi <= pix_lo or hi <= lo:
            return lo
        frac = (pixel - pix_lo) / (pix_hi - pix_lo)
        return lo + frac * (hi - lo)

    def _ensure_pixels(self) -> None:
        """Cache data→pixel mapping for the current widget size."""
        rect = self._plot_rect()
        self._x_range = self._padded_range(self._xs)
        self._y_range = self._padded_range(self._ys)
        n = self._xs.size
        px = np.empty((n, 2), dtype=np.float64)
        if n:
            x_lo, x_hi = self._x_range
            y_lo, y_hi = self._y_range
            xs = np.interp(self._xs, (x_lo, x_hi),
                           (rect.left(), rect.right()))
            ys = np.interp(self._ys, (y_lo, y_hi),
                           (rect.bottom(), rect.top()))
            px[:, 0], px[:, 1] = xs, ys
        self._px = px

    # ---- painting -----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        palette = self.palette()
        text_color = palette.color(palette.ColorRole.WindowText)
        base_color = palette.color(palette.ColorRole.Base)
        mid_color = palette.color(palette.ColorRole.PlaceholderText)

        painter.fillRect(self.rect(), base_color)
        rect = self._plot_rect()

        # grid + ticks
        painter.setFont(self.font())
        grid_pen = QPen(mid_color)
        grid_pen.setWidthF(0.6)
        painter.setPen(grid_pen)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.left() + frac * rect.width()
            y = rect.bottom() - frac * rect.height()
            painter.drawLine(QPointF(x, rect.top()),
                             QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y),
                             QPointF(rect.right(), y))
            tick_pen = QPen(text_color)
            tick_pen.setWidthF(0.8)
            painter.setPen(tick_pen)
            x_val = self._pixel_to_value(x, *self._x_range,
                                         rect.left(), rect.right())
            y_val = self._pixel_to_value(y, *self._y_range,
                                         rect.left(), rect.right())
            painter.drawText(int(x) - 24, rect.bottom() + 18, 48, 14,
                             Qt.AlignmentFlag.AlignHCenter,
                             _format_tick(x_val))
            painter.drawText(2, int(y) - 7, _MARGIN_LEFT - 8, 14,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             _format_tick(y_val))
            painter.setPen(grid_pen)

        # axis labels
        label_pen = QPen(text_color)
        painter.setPen(label_pen)
        painter.drawText(rect.left(), rect.bottom() + 22, rect.width(), 16,
                         Qt.AlignmentFlag.AlignCenter, self._x_label)
        painter.save()
        painter.translate(14, rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, -8, rect.height(), 16,
                         Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()

        # points
        if self._xs.size:
            if self._px is None:
                self._ensure_pixels()
            # Polygon building dominates paint time for large N — cache the
            # QPolygons per widget size and invalidate on resize/set_plot.
            if self._polys is None or self._poly_size != (self.width(),
                                                          self.height()):
                self._build_polygons()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            for color_index, poly in self._polys.items():
                color = QColor(_POINT_COLORS[color_index % len(_POINT_COLORS)])
                color.setAlpha(170)
                painter.setPen(QPen(color, 3.0,
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawPoints(poly)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if not self._xs.size:
            painter.setPen(QPen(mid_color))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             self._status or "No data — pick sources and plot")
        elif self._status:
            painter.setPen(QPen(mid_color))
            painter.drawText(self.rect().adjusted(_MARGIN_LEFT, 0,
                                                  -_MARGIN_RIGHT,
                                                  -_MARGIN_BOTTOM + 14),
                             Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignBottom,
                             self._status)

    def _build_polygons(self) -> None:
        """Group points by colour and build one QPolygonF per group."""
        self._polys = {}
        many_groups = (self._colors
                       and len(self._colors) == self._xs.size
                       and len(set(self._colors)) > _MAX_COLOR_GROUPS)
        if not self._colors or len(self._colors) != self._xs.size \
                or many_groups:
            # Single group: the palette's first colour, all points.
            poly = QPolygonF(list(map(QPointF, self._px[:, 0].tolist(),
                                      self._px[:, 1].tolist())))
            self._polys[0] = poly
        else:
            colors = np.asarray(self._colors)
            for color_index in sorted(set(self._colors)):
                pts = self._px[colors == color_index]
                self._polys[color_index] = QPolygonF(
                    list(map(QPointF, pts[:, 0].tolist(), pts[:, 1].tolist())))
        self._poly_size = (self.width(), self.height())

    # ---- interaction --------------------------------------------------------
    def _nearest_point(self, x: float, y: float) -> int:
        """Index of the plotted point nearest to *(x, y)* pixel coords, or
        ``-1`` when nothing lies within :data:`_PICK_RADIUS`."""
        if not self._xs.size or self._px is None:
            return -1
        deltas = self._px - np.array([x, y])
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] <= _PICK_RADIUS else -1

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            index = self._nearest_point(pos.x(), pos.y())
            if index >= 0:
                self.point_clicked.emit(index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._xs.size and self._px is not None:
            pos = event.position()
            deltas = self._px - np.array([pos.x(), pos.y()])
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            nearest = int(np.argmin(distances))
            if distances[nearest] <= _PICK_RADIUS:
                label = (self._labels[nearest]
                         if nearest < len(self._labels)
                         else f"point {nearest}")
                QToolTip.showText(event.globalPosition().toPoint(), label,
                                  self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)
