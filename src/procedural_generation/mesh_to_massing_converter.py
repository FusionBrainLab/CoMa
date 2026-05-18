"""Convert trimesh meshes to massing format used in the dataset."""
from .mesh_utils import extract_footprint_at_height


class MeshToMassingConverter:
    """Converts trimesh meshes to the massing format for different strategies."""
    
    def convert_simple(self, mesh, building_id="0", footprint=None):
        """
        Convert a simple extruded mesh to massing format.
        Creates a SINGLE extrusion from bottom to top using the ORIGINAL footprint.
        
        Args:
            mesh: trimesh.Trimesh object (simple extrusion)
            building_id: ID string for the building
            footprint: Original footprint polygon (if None, extracts from mesh)
        
        Returns:
            Dict with "id" and "massing" fields in dataset format
        """
        if len(mesh.vertices) == 0:
            return {"id": building_id, "massing": []}
        
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        
        # Use original footprint if provided (NO simplification or modification)
        if footprint is not None:
            polygon = [[float(p[0]), float(p[1])] for p in footprint]
        else:
            # Extract footprint polygon from mesh as fallback
            polygon = extract_footprint_at_height(mesh, z_min, z_max)
        
        return {
            "id": building_id,
            "massing": [{
                "polygons": [polygon],
                "bottom_elevation": float(z_min),
                "top_elevation": float(z_max)
            }]
        }
    
    def convert_setback(self, mesh, building_id="0", num_setbacks=3, footprint=None):
        """
        Convert a setback mesh to massing format.
        Creates multiple levels with progressive setbacks (pyramid style).
        Each level is 15% smaller than the previous one.
        
        Args:
            mesh: trimesh.Trimesh object (setback design)
            building_id: ID string for the building
            num_setbacks: Number of setback levels (default: 3)
            footprint: Original footprint polygon (REQUIRED for proper setback calculation)
        
        Returns:
            Dict with "id" and "massing" fields in dataset format
        """
        if len(mesh.vertices) == 0:
            return {"id": building_id, "massing": []}
        
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        total_height = z_max - z_min
        height_per_level = total_height / num_setbacks
        
        massing_elements = []
        
        # If footprint provided, compute scaled versions; otherwise extract from mesh
        if footprint is not None:
            from shapely.geometry import Polygon
            from shapely import affinity
            
            base_poly = Polygon(footprint)
            
            for level in range(num_setbacks):
                z_bottom = z_min + level * height_per_level
                z_top = z_min + (level + 1) * height_per_level
                
                # Scale the footprint: level 0 = 100%, level 1 = 85%, level 2 = 70%
                inset_ratio = 1.0 - (level * 0.15)
                level_poly = affinity.scale(
                    base_poly, xfact=inset_ratio, yfact=inset_ratio, origin="centroid"
                )
                
                # Extract coordinates
                polygon = [[float(p[0]), float(p[1])] for p in level_poly.exterior.coords[:-1]]
                
                massing_elements.append({
                    "polygons": [polygon],
                    "bottom_elevation": float(z_bottom),
                    "top_elevation": float(z_top)
                })
        else:
            # Fallback: extract from mesh
            for level in range(num_setbacks):
                z_bottom = z_min + level * height_per_level
                z_top = z_min + (level + 1) * height_per_level
                z_mid = (z_bottom + z_top) / 2
                
                polygon = extract_footprint_at_height(mesh, z_mid - 0.05, z_mid + 0.05, tolerance=0.01)
                
                massing_elements.append({
                    "polygons": [polygon],
                    "bottom_elevation": float(z_bottom),
                    "top_elevation": float(z_top)
                })
        
        return {
            "id": building_id,
            "massing": massing_elements
        }
    
    def convert_podium_tower(self, mesh, building_id="0", footprint=None):
        """
        Convert a podium-tower mesh to massing format.
        Creates two parts: podium (30% height, 100% footprint) and tower (70% height, 60% footprint).
        
        Args:
            mesh: trimesh.Trimesh object (podium + tower design)
            building_id: ID string for the building
            footprint: Original footprint polygon (REQUIRED for proper scaling)
        
        Returns:
            Dict with "id" and "massing" fields in dataset format
        """
        if len(mesh.vertices) == 0:
            return {"id": building_id, "massing": []}
        
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        total_height = z_max - z_min
        
        # Podium: 30% of height
        podium_height = total_height * 0.3
        z_podium_top = z_min + podium_height
        
        if footprint is not None:
            from shapely.geometry import Polygon
            from shapely import affinity
            
            base_poly = Polygon(footprint)
            
            # Podium: 100% footprint
            podium_polygon = [[float(p[0]), float(p[1])] for p in footprint]
            
            # Tower: 60% footprint (scaled from center)
            tower_poly = affinity.scale(base_poly, xfact=0.6, yfact=0.6, origin="centroid")
            tower_polygon = [[float(p[0]), float(p[1])] for p in tower_poly.exterior.coords[:-1]]
        else:
            # Fallback: extract from mesh
            podium_polygon = extract_footprint_at_height(mesh, z_min, z_min + 0.1)
            tower_polygon = extract_footprint_at_height(mesh, z_max - 0.1, z_max)
        
        massing_elements = [
            {
                "polygons": [podium_polygon],
                "bottom_elevation": float(z_min),
                "top_elevation": float(z_podium_top)
            },
            {
                "polygons": [tower_polygon],
                "bottom_elevation": float(z_podium_top),
                "top_elevation": float(z_max)
            }
        ]
        
        return {
            "id": building_id,
            "massing": massing_elements
        }
    
    def convert_varied(self, mesh, building_id="0", footprint=None):
        """
        Convert a varied height mesh to massing format.
        Creates 3 segments with heights 40%/30%/30% and widths 100%/85%/70%.
        
        Args:
            mesh: trimesh.Trimesh object (varied design)
            building_id: ID string for the building
            footprint: Original footprint polygon (REQUIRED for proper scaling)
        
        Returns:
            Dict with "id" and "massing" fields in dataset format
        """
        if len(mesh.vertices) == 0:
            return {"id": building_id, "massing": []}
        
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        total_height = z_max - z_min
        
        # Height distribution: 40%, 30%, 30%
        heights = [total_height * 0.4, total_height * 0.3, total_height * 0.3]
        # Width distribution: 100%, 85%, 70%
        widths = [1.0, 0.85, 0.7]
        
        massing_elements = []
        current_z = z_min
        
        if footprint is not None:
            from shapely.geometry import Polygon
            from shapely import affinity
            
            base_poly = Polygon(footprint)
            
            for i, (h, w) in enumerate(zip(heights, widths)):
                z_bottom = current_z
                z_top = current_z + h
                
                # Scale footprint according to width ratio
                if w == 1.0:
                    polygon = [[float(p[0]), float(p[1])] for p in footprint]
                else:
                    scaled_poly = affinity.scale(base_poly, xfact=w, yfact=w, origin="centroid")
                    polygon = [[float(p[0]), float(p[1])] for p in scaled_poly.exterior.coords[:-1]]
                
                massing_elements.append({
                    "polygons": [polygon],
                    "bottom_elevation": float(z_bottom),
                    "top_elevation": float(z_top)
                })
                
                current_z = z_top
        else:
            # Fallback: extract from mesh
            for i, h in enumerate(heights):
                z_bottom = current_z
                z_top = current_z + h
                z_mid = (z_bottom + z_top) / 2
                
                polygon = extract_footprint_at_height(mesh, z_mid - 0.1, z_mid + 0.1)
                
                massing_elements.append({
                    "polygons": [polygon],
                    "bottom_elevation": float(z_bottom),
                    "top_elevation": float(z_top)
                })
                
                current_z = z_top
        
        return {
            "id": building_id,
            "massing": massing_elements
        }
    
    def convert(self, mesh, building_id="0", strategy="simple", footprint=None):
        """
        Convert a trimesh mesh to the massing format used in the dataset.
        Uses strategy-specific conversion logic to preserve the design intent.
        
        Args:
            mesh: trimesh.Trimesh object
            building_id: ID string for the building
            strategy: Generation strategy used ("simple", "setback", "podium_tower", "varied")
            footprint: Original footprint polygon (optional, preserves exact footprint shape)
            
        Returns:
            Dict with "id" and "massing" fields in dataset format
        """
        if strategy == "simple":
            return self.convert_simple(mesh, building_id, footprint=footprint)
        elif strategy == "setback":
            return self.convert_setback(mesh, building_id, num_setbacks=3, footprint=footprint)
        elif strategy == "podium_tower":
            return self.convert_podium_tower(mesh, building_id, footprint=footprint)
        elif strategy == "varied":
            return self.convert_varied(mesh, building_id, footprint=footprint)
        else:
            # Default to simple
            return self.convert_simple(mesh, building_id, footprint=footprint)
