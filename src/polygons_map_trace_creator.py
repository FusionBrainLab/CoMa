from typing import Any, List, Tuple

from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go

from .core.base import Function

class PolygonsMapTraceCreator(Function):
    def __init__(self, *, fill_color: str,
                        line_color: str) -> None:
        self.fill_color = fill_color
        self.line_color = line_color
        
    def __call__(self, *, polygons: List[List[Tuple[float, float]]]) -> List[BaseTraceType]:
        traces = []
        for i, polygon in enumerate(polygons):
            lons = [p[0] for p in polygon] + [polygon[0][0]]
            lats = [p[1] for p in polygon] + [polygon[0][1]]
            traces.append(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                fill='toself',
                fillcolor=self.fill_color,
                line=dict(width=2, color=self.line_color)
            ))
        return traces