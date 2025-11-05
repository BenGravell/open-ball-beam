"""Utilities for working with Qt (PySide6)."""

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg  # type: ignore[import]


def center_qt_window(win: pg.GraphicsLayoutWidget) -> None:
    """Center a Qt window on the active screen."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication instance must be running.")

    screen = app.primaryScreen()
    cursor_pos = QtGui.QCursor.pos()
    for s in app.screens():
        if s.geometry().contains(cursor_pos):
            screen = s
            break

    frame_geometry = win.frameGeometry()
    center_point = screen.geometry().center()
    frame_geometry.moveCenter(center_point)
    win.move(frame_geometry.topLeft())


def setup_legend(plot: pg.PlotItem) -> None:
    """Create legend in plot and apply style settings."""
    legend = plot.addLegend(pen=pg.mkPen(color="w", width=1, style=QtCore.Qt.SolidLine))
    legend_label_style = {"size": "10pt", "color": "w"}

    for item in legend.items:
        for single_item in item:
            if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                single_item.setText(single_item.text, **legend_label_style)

    # Set legend background color
    legend.setBrush((64, 64, 64, 224))
