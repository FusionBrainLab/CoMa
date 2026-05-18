from typing import List, Dict, Any, Optional
import math
import os

import pyvista as pv
import trimesh
from PIL import Image
import numpy as np

from .image_data_visualizer import ImageDataVisualizer


class ContextualMeshVisualizer(ImageDataVisualizer):
    """
    Visualizer for contextual mesh rendering.
    
    Renders a target mesh in context with surrounding/environment meshes.
    Applies transform (base_point) to meshes and selects the best camera angle.
    
    Input data format:
    {
        "pred_meshes": [{"id": "1", "massing": "path/to/mesh.ply"}, ...],  # target mesh(es)
        "context_meshes": [{"massing": "path/to/context1.ply"}, ...],  # context mesh(es), optional
        "base_point": [x, y],  # XY transform to apply, optional
        "massing_color": "red",  # color for target mesh, optional (default "red")
        "context_color": "grey",  # color for context meshes, optional (default "grey")
    }
    
    For TRELLIS evaluation compatibility:
    {
        "pred_meshes": [{"massing": "path/to/mesh.ply"}],  # target mesh(es)
        "base_point": [x, y],  # transform
    }
    """
    
    def __init__(
        self,
        *,
        massing_color: str = "red",
        context_color: str = "grey",
        window_size: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the contextual mesh visualizer.
        
        Args:
            massing_color: Color for the target mesh
            context_color: Color for context/environment meshes
            window_size: Optional window size [width, height]
        """
        self.massing_color = massing_color
        self.context_color = context_color
        self.window_size = window_size or [800, 600]
        
        pv.global_theme.allow_empty_mesh = True
    
    def _polygon_to_trimesh(self, massing_dict: dict) -> Optional[trimesh.Trimesh]:
        """Convert polygon geometry dict to trimesh."""
        try:
            verts: List[tuple] = []
            faces: List[tuple] = []
            vert_offset = 0
            
            for extrusion in massing_dict.get("massing", []):
                polygons = extrusion.get("polygons", [])
                bottom = float(extrusion.get("bottom_elevation", 0.0))
                top = float(extrusion.get("top_elevation", 1.0))
                
                for poly in polygons:
                    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
                        continue
                    
                    ring = [list(map(float, p)) for p in poly]
                    n = len(ring)
                    
                    # Add bottom verts then top verts
                    for (x, y) in ring:
                        verts.append([x, y, bottom])
                    for (x, y) in ring:
                        verts.append([x, y, top])
                    
                    # Bottom face (fan triangulation)
                    for i in range(1, n - 1):
                        faces.append([vert_offset + 0, vert_offset + i + 1, vert_offset + i])
                    
                    # Top face (reverse order so normal points up)
                    top_offset = vert_offset + n
                    for i in range(1, n - 1):
                        faces.append([top_offset + 0, top_offset + i, top_offset + i + 1])
                    
                    # Sides
                    for i in range(n):
                        j = (i + 1) % n
                        b_i = vert_offset + i
                        b_j = vert_offset + j
                        t_i = vert_offset + n + i
                        t_j = vert_offset + n + j
                        faces.append([b_i, b_j, t_j])
                        faces.append([b_i, t_j, t_i])
                    
                    vert_offset += 2 * n
            
            if not verts or not faces:
                return None
            
            return trimesh.Trimesh(
                vertices=np.asarray(verts, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False
            )
        except Exception as e:
            print(f"Error converting polygon geometry to mesh: {e}")
            return None

    def _load_mesh(self, mesh_input: str | dict) -> Optional[trimesh.Trimesh]:
        """Load a single mesh from file path or polygon geometry dict."""
        # Handle polygon geometry dict
        if isinstance(mesh_input, dict) and "massing" in mesh_input:
            return self._polygon_to_trimesh(mesh_input)
        
        # Handle file path
        mesh_path = str(mesh_input)
        try:
            return trimesh.load(mesh_path)
        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            return None
    
    def _merge_meshes(self, mesh_list: List[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
        """Merge multiple meshes into one."""
        if not mesh_list:
            return None
        
        valid_meshes = [m for m in mesh_list if m is not None]
        if not valid_meshes:
            return None
        
        if len(valid_meshes) == 1:
            return valid_meshes[0]
        
        # Try union, fall back to concatenation
        merged = valid_meshes[0]
        for mesh in valid_meshes[1:]:
            try:
                merged = merged.union(mesh)
            except Exception:
                try:
                    merged = trimesh.util.concatenate([merged, mesh])
                except Exception as e:
                    print(f"Failed to merge meshes: {e}")
                    continue
        
        return merged
    
    def _apply_transform(
        self, mesh: trimesh.Trimesh, base_point: Optional[List[float]]
    ) -> trimesh.Trimesh:
        """Apply XY translation transform to mesh."""
        if base_point is None or len(base_point) < 2:
            return mesh
        
        # Apply XY translation, Z is 0
        translation = np.array([base_point[0], base_point[1], 0.0])
        mesh.vertices += translation
        return mesh
    
    def _get_best_view_image(
        self,
        plotter: pv.Plotter,
        massing_mesh: trimesh.Trimesh,
        camera_distance: float,
    ) -> Image.Image:
        """
        Render multiple camera angles and select the one with maximum target mesh visibility.
        
        Uses red pixel count (target mesh color) to determine the best angle.
        """
        massing_vertices = massing_mesh.vertices
        x_min, x_max = np.min(massing_vertices[:, 0]), np.max(massing_vertices[:, 0])
        y_min, y_max = np.min(massing_vertices[:, 1]), np.max(massing_vertices[:, 1])
        z_min, z_max = np.min(massing_vertices[:, 2]), np.max(massing_vertices[:, 2])
        
        massing_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2,
        ])
        
        # Camera angles: 4 isometric views around the object
        xs = [0, -1, 0, 1]
        ys = [-1, 0, 1, 0]
        
        max_red_percentage = 0
        best_image = None
        fallback_image = None
        
        with pv.vtk_verbosity('off'):
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                
                # Isometric camera position
                camera_pos = (
                    massing_center[0] + x * camera_distance * math.cos(math.pi / 4),
                    massing_center[1] + y * camera_distance * math.cos(math.pi / 4),
                    massing_center[2] + camera_distance * math.sin(math.pi / 4),
                )
                
                plotter.set_viewup((0.0, 0.0, 1.0))
                plotter.set_position(camera_pos)
                plotter.set_focus(massing_center)
                
                # Render screenshot
                image_array = plotter.screenshot(return_img=True)
                np_image = np.array(image_array)
                image = Image.fromarray(np_image)
                
                # Save as fallback
                if fallback_image is None:
                    fallback_image = image
                
                # Check red pixel percentage (target mesh color)
                red_lower = np.array([200, 0, 0])
                red_upper = np.array([255, 100, 100])
                red_mask = np.all((np_image >= red_lower) & (np_image <= red_upper), axis=-1)
                red_pixel_count = np.sum(red_mask)
                total_pixels = np_image.shape[0] * np_image.shape[1]
                red_percentage = red_pixel_count / total_pixels
                
                if red_percentage > max_red_percentage:
                    max_red_percentage = red_percentage
                    best_image = image
        
        return best_image if best_image is not None else fallback_image
    
    def __call__(self, *, data: Dict[str, Any]) -> Image.Image:
        """
        Visualize the contextual mesh.
        
        Args:
            data: Input data dictionary with "pred_meshes" and optional "context_meshes", "base_point"
        
        Returns:
            PIL Image of the rendered visualization
        """
        # Extract target meshes
        pred_meshes_data = data.get("pred_meshes", [])
        if not pred_meshes_data:
            raise ValueError("No 'pred_meshes' in input data")
        
        # Load target meshes
        pred_meshes = []
        for mesh_data in pred_meshes_data:
            if isinstance(mesh_data, dict) and "massing" in mesh_data:
                mesh_path = mesh_data["massing"]
            else:
                mesh_path = str(mesh_data)
            
            mesh = self._load_mesh(mesh_path)
            if mesh is not None:
                pred_meshes.append(mesh)
        
        if not pred_meshes:
            raise ValueError("Failed to load any target meshes")
        
        # Merge target meshes
        massing_mesh = self._merge_meshes(pred_meshes)
        if massing_mesh is None:
            raise ValueError("Failed to merge target meshes")
        
        # Apply transform (base_point)
        base_point = data.get("base_point")
        if base_point is not None:
            massing_mesh = self._apply_transform(massing_mesh, base_point)
        
        # Load context meshes (optional)
        context_meshes = []
        context_meshes_data = data.get("context_meshes", [])
        for mesh_data in context_meshes_data:
            # mesh_data can be:
            # 1. A dict with polygon geometry: {"id": ..., "massing": [...]}
            # 2. A dict with file path: {"massing": "path/to/file.ply"}
            # 3. A string (file path)
            mesh = self._load_mesh(mesh_data)
            if mesh is not None:
                context_meshes.append(mesh)
        
        # Merge context meshes
        context_mesh = self._merge_meshes(context_meshes) if context_meshes else None
        
        # Get colors (can be overridden per call)
        massing_color = data.get("massing_color", self.massing_color)
        context_color = data.get("context_color", self.context_color)
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Add target mesh
        plotter.add_mesh(
            massing_mesh,
            color=massing_color,
            opacity=1.0,
            show_edges=False,
        )
        
        # Add context meshes if available
        if context_mesh is not None:
            plotter.add_mesh(
                context_mesh,
                color=context_color,
                opacity=1.0,
                show_edges=False,
            )
        
        plotter.reset_camera()
        
        # Calculate camera distance from mesh bounds
        massing_vertices = massing_mesh.vertices
        x_min, x_max = np.min(massing_vertices[:, 0]), np.max(massing_vertices[:, 0])
        y_min, y_max = np.min(massing_vertices[:, 1]), np.max(massing_vertices[:, 1])
        z_min, z_max = np.min(massing_vertices[:, 2]), np.max(massing_vertices[:, 2])
        
        massing_width = x_max - x_min
        massing_length = y_max - y_min
        massing_height = z_max - z_min
        camera_distance = 5 * max(massing_width, massing_length, massing_height)
        
        # Get the best view
        image = self._get_best_view_image(plotter, massing_mesh, camera_distance)
        
        return image
