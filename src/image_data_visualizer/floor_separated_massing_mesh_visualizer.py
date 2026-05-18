from typing import Any, Dict, List

from PIL import Image
import numpy as np
import pyvista as pv
import trimesh

from .image_data_visualizer import ImageDataVisualizer

def split_roof_and_volume(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    roof_face_mask = mesh.face_normals[:, 2] >= 0.95
    roof_faces = np.nonzero(roof_face_mask)[0]
    volume_faces = np.nonzero(~roof_face_mask)[0]

    roof_mesh = mesh.submesh([roof_faces], append=True, repair=False) if len(roof_faces) else None
    volume_mesh = mesh.submesh([volume_faces], append=True, repair=False) if len(volume_faces) else None
    return roof_mesh, volume_mesh

def get_floor_heights(mesh: trimesh.Trimesh, floor_height: float) -> np.ndarray:
    z_min = float(np.min(mesh.vertices[:, 2]))
    z_max = float(np.max(mesh.vertices[:, 2]))
    if floor_height <= 0:
        raise ValueError("floor_height must be positive")

    heights = np.arange(z_min + floor_height, z_max, floor_height)
    return heights[heights < z_max - 1e-6]

def section_to_polydata(section: Any) -> pv.PolyData:
    points: list[np.ndarray] = []
    lines: list[int] = []
    point_offset = 0

    for polyline in section.discrete:
        if len(polyline) < 2:
            continue

        points.extend(polyline)
        for i in range(len(polyline) - 1):
            lines.extend([2, point_offset + i, point_offset + i + 1])
        point_offset += len(polyline)

    if not points:
        return None

    polydata = pv.PolyData(np.asarray(points))
    polydata.lines = np.asarray(lines)
    return polydata

def add_floor_lines(plotter: pv.Plotter, 
                    mesh: trimesh.Trimesh, 
                    floor_heights: np.ndarray,
                    floor_line_color: str,
                    floor_line_width: int) -> None:
    for height in floor_heights:
        section = mesh.section(
            plane_origin=[0.0, 0.0, float(height)],
            plane_normal=[0.0, 0.0, 1.0],
        )
        polydata = section_to_polydata(section)

        plotter.add_mesh(
            polydata,
            color=floor_line_color,
            line_width=floor_line_width,
        )

class FloorSeparatedMassingMeshVisualizer(ImageDataVisualizer):
    def __init__(self, *, window_size: List[int],
                        roof_color: str,
                        volume_color: str,
                        floor_line_color: str,
                        floor_height: float,
                        angle: int,
                        massing_mesh_key: str,
                        floor_line_width: int = 3) -> None:
        self.window_size = window_size
        self.roof_color = roof_color
        self.volume_color = volume_color
        self.floor_line_color = floor_line_color
        self.floor_height = floor_height
        self.angle = angle
        self.massing_mesh_key = massing_mesh_key
        self.floor_line_width = floor_line_width

        pv.global_theme.allow_empty_mesh = True

    def __call__(self, *, data: Dict[str, Any]) -> Image:
        massing_mesh = data[self.massing_mesh_key]
        roof_mesh, volume_mesh = split_roof_and_volume(massing_mesh)

        with pv.vtk_verbosity("off"):
            plotter = pv.Plotter(
                off_screen=True,
                window_size=self.window_size,
            )

            if volume_mesh is not None:
                plotter.add_mesh(
                    volume_mesh,
                    color=self.volume_color,
                    opacity=1,
                    show_edges=False,
                )

            if roof_mesh is not None:
                plotter.add_mesh(
                    roof_mesh,
                    color=self.roof_color,
                    opacity=1,
                    show_edges=False,
                )

            floor_heights = get_floor_heights(massing_mesh, self.floor_height)
            add_floor_lines(plotter, massing_mesh, floor_heights, self.floor_line_color, self.floor_line_width)

            plotter.view_isometric()
            plotter.camera.azimuth = self.angle
            plotter.camera.up = (0, 0, 1)
            plotter.reset_camera_clipping_range()
            image = plotter.screenshot(return_img=True)

            np_image = np.array(image)
            return Image.fromarray(np_image)
