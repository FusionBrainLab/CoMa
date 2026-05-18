"""Utility functions for mesh processing and polygon manipulation."""
import numpy as np


def simplify_polygon(points, max_points=20, tolerance=0.5):
    """
    Simplify a polygon by reducing the number of points using Douglas-Peucker algorithm.
    
    Args:
        points: List of [x, y] coordinates
        max_points: Maximum number of points to retain
        tolerance: Simplification tolerance for Douglas-Peucker algorithm
    
    Returns:
        Simplified list of [x, y] coordinates
    """
    from shapely.geometry import Polygon
    from shapely import simplify as shapely_simplify
    
    if len(points) <= max_points:
        return points
    
    try:
        # Create shapely polygon
        poly = Polygon(points)
        
        # Simplify using Douglas-Peucker algorithm
        simplified = shapely_simplify(poly, tolerance=tolerance, preserve_topology=True)
        
        # Extract coordinates
        if hasattr(simplified, 'exterior'):
            coords = list(simplified.exterior.coords[:-1])  # Remove duplicate last point
            if len(coords) >= 3:
                return [[float(p[0]), float(p[1])] for p in coords]
    except:
        pass
    
    # Fallback: sample points evenly
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return [points[i] for i in indices]
    
    return points


def extract_footprint_at_height(mesh, z_min, z_max, tolerance=0.1):
    """
    Extract 2D footprint polygon from mesh vertices within a height range.
    
    Args:
        mesh: trimesh.Trimesh object
        z_min: Minimum Z coordinate
        z_max: Maximum Z coordinate
        tolerance: Tolerance for Z-coordinate matching
        
    Returns:
        List of [x, y] coordinates representing the footprint polygon
    """
    from scipy.spatial import ConvexHull
    
    # Get vertices within the height range (with tolerance)
    mask = (mesh.vertices[:, 2] >= z_min - tolerance) & (mesh.vertices[:, 2] <= z_max + tolerance)
    vertices_in_range = mesh.vertices[mask]
    
    if len(vertices_in_range) < 3:
        # If not enough vertices, use all vertices
        vertices_in_range = mesh.vertices
    
    # Project to 2D
    vertices_2d = vertices_in_range[:, :2]
    
    try:
        # Use convex hull to get the footprint
        hull = ConvexHull(vertices_2d)
        polygon = [[float(vertices_2d[idx, 0]), float(vertices_2d[idx, 1])] 
                  for idx in hull.vertices]
        return polygon
    except:
        # Fallback: bounding box
        x_min, y_min = vertices_2d.min(axis=0)
        x_max, y_max = vertices_2d.max(axis=0)
        return [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)]
        ]


def scale_polygon_to_area(polygon, target_area):
    """
    Scale a polygon to achieve a target area.
    
    Args:
        polygon: List of [x, y] coordinates
        target_area: Desired area in square meters
    
    Returns:
        Scaled polygon as list of [x, y] coordinates
    """
    from shapely.geometry import Polygon
    from shapely import affinity
    
    try:
        poly = Polygon(polygon)
        current_area = poly.area
        
        if current_area <= 0:
            return polygon
        
        # Calculate scale factor (area scales with square of linear scale)
        scale_factor = np.sqrt(target_area / current_area)
        
        # Scale polygon from its centroid
        scaled_poly = affinity.scale(poly, xfact=scale_factor, yfact=scale_factor, origin="centroid")
        
        # Extract coordinates
        return [[float(p[0]), float(p[1])] for p in scaled_poly.exterior.coords[:-1]]
    except Exception as e:
        print(f"Error scaling polygon: {e}")
        return polygon


def calculate_polygon_area(polygon):
    """
    Calculate the area of a polygon.
    
    Args:
        polygon: List of [x, y] coordinates
    
    Returns:
        Area in square meters
    """
    from shapely.geometry import Polygon
    
    try:
        poly = Polygon(polygon)
        return poly.area
    except:
        return 0.0
