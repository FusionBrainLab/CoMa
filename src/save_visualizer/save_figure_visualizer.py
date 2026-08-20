from typing import Any, Dict, List
import os

import matplotlib
import matplotlib.pyplot as plt

from .save_visualizer import SaveVisualizer
from ..figure_data_visualizer import FigureDataVisualizer

class SaveFigureVisualizer(SaveVisualizer):
    def __init__(self, *, figure_visualizer: FigureDataVisualizer,
                        path: str,
                        formats: List[str],
                        dpi: float,
                        font_family: str,
                        pgf_texsystem: str,
                        pgf_preamble: List[str]) -> None:
        self.figure_visualizer = figure_visualizer
        self.path = path
        self.formats = formats
        self.dpi = dpi
        self.font_family = font_family
        self.pgf_texsystem = pgf_texsystem
        self.pgf_preamble = pgf_preamble

    def __call__(self, *, data: Dict[str, Any]) -> str:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for figure_format in self.formats:
            rc_params = {
                "font.family": self.font_family,
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "axes.unicode_minus": False,
            }
            if figure_format == "pgf":
                rc_params["pgf.texsystem"] = self.pgf_texsystem
                rc_params["pgf.rcfonts"] = False
                rc_params["pgf.preamble"] = "\n".join(self.pgf_preamble)
            # Rebuilt per format because text extents are measured by the renderer that
            # writes the file: LaTeX for pgf, the font backend otherwise.
            with matplotlib.rc_context(rc_params):
                figure = self.figure_visualizer(data=data)
                figure.savefig(f"{self.path}.{figure_format}", format=figure_format, dpi=self.dpi)
                plt.close(figure)
        return self.path
