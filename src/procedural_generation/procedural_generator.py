"""Generate procedural building massings from requirements and site contour."""
import trimesh
from .parameter_extractor import ParameterExtractor


class ProceduralMassingGenerator:
    """Generates procedural building massings from requirements and site contour."""
    
    def __init__(self):
        self.parameter_extractor = ParameterExtractor()
        
    def create_extruded_mesh(self, footprint_polygon, height):
        """
        SIMPLE strategy: Basic vertical extrusion of footprint to full height.
        
        Args:
            footprint_polygon: List of [x, y] coordinates
            height: Total height in meters
        
        Returns:
            trimesh.Trimesh object
        """
        if height <= 0:
            height = 10.0

        from shapely.geometry import Polygon

        try:
            poly = Polygon(footprint_polygon)
            mesh = trimesh.creation.extrude_polygon(poly, height=height)
            return mesh
        except Exception as e:
            print(f"Error creating extruded mesh: {e}")
            return trimesh.creation.box(extents=[10, 10, height])

    def create_setback_mesh(self, footprint_polygon, height, num_setbacks=3):
        """
        SETBACK strategy: Terraced building with progressive setbacks.
        
        Args:
            footprint_polygon: List of [x, y] coordinates
            height: Total height in meters
            num_setbacks: Number of setback levels (default: 3)
        
        Returns:
            trimesh.Trimesh object with setbacks
        """
        if height <= 0:
            height = 10.0

        from shapely.geometry import Polygon
        from shapely import affinity

        try:
            meshes = []
            base_poly = Polygon(footprint_polygon)
            height_per_level = height / num_setbacks

            for level in range(num_setbacks):
                inset_ratio = 1.0 - (level * 0.15)  # 15% reduction per level
                level_poly = affinity.scale(
                    base_poly, xfact=inset_ratio, yfact=inset_ratio, origin="centroid"
                )
                level_mesh = trimesh.creation.extrude_polygon(
                    level_poly, height=height_per_level
                )
                level_mesh.vertices[:, 2] += level * height_per_level
                meshes.append(level_mesh)

            return trimesh.util.concatenate(meshes)

        except Exception as e:
            print(f"Error creating setback mesh: {e}")
            return self.create_extruded_mesh(footprint_polygon, height)

    def create_podium_tower_mesh(self, footprint_polygon, height):
        """
        PODIUM_TOWER strategy: Wide base with narrow tower.
        
        Args:
            footprint_polygon: List of [x, y] coordinates
            height: Total height in meters
        
        Returns:
            trimesh.Trimesh object with podium and tower
        """
        if height <= 0:
            height = 10.0

        from shapely.geometry import Polygon
        from shapely import affinity

        try:
            base_poly = Polygon(footprint_polygon)

            # Podium: 30% height, full footprint
            podium_height = height * 0.3
            podium_mesh = trimesh.creation.extrude_polygon(
                base_poly, height=podium_height
            )

            # Tower: 70% height, 60% footprint
            tower_height = height * 0.7
            tower_poly = affinity.scale(
                base_poly, xfact=0.6, yfact=0.6, origin="centroid"
            )
            tower_mesh = trimesh.creation.extrude_polygon(
                tower_poly, height=tower_height
            )
            tower_mesh.vertices[:, 2] += podium_height

            return trimesh.util.concatenate([podium_mesh, tower_mesh])

        except Exception as e:
            print(f"Error creating podium-tower mesh: {e}")
            return self.create_extruded_mesh(footprint_polygon, height)

    def create_varied_height_mesh(self, footprint_polygon, height):
        """
        VARIED strategy: 3 segments with different heights and progressive narrowing.
        
        Args:
            footprint_polygon: List of [x, y] coordinates
            height: Total height in meters
        
        Returns:
            trimesh.Trimesh object with varied segments
        """
        if height <= 0:
            height = 10.0

        from shapely.geometry import Polygon
        from shapely import affinity

        try:
            base_poly = Polygon(footprint_polygon)
            heights = [height * 0.4, height * 0.3, height * 0.3]  # Height distribution
            widths = [1.0, 0.85, 0.7]  # Progressive narrowing

            meshes = []
            current_height = 0

            for h, w in zip(heights, widths):
                segment_poly = affinity.scale(
                    base_poly, xfact=w, yfact=w, origin="centroid"
                )
                segment_mesh = trimesh.creation.extrude_polygon(segment_poly, height=h)
                segment_mesh.vertices[:, 2] += current_height
                meshes.append(segment_mesh)
                current_height += h

            return trimesh.util.concatenate(meshes)

        except Exception as e:
            print(f"Error creating varied height mesh: {e}")
            return self.create_extruded_mesh(footprint_polygon, height)

    def generate_massing(self, requirements, site_contour, strategy="simple"):
        """
        Generate massing using specified strategy.
        
        Now properly handles usable area by scaling footprint to match:
        footprint_area = usable_area / n_floors
        
        Args:
            requirements: List of requirement dicts (or single requirement)
            site_contour: Site contour polygon
            strategy: "simple", "setback", "podium_tower", or "varied"
        
        Returns:
            trimesh.Trimesh object
        """
        # Handle single requirement
        if isinstance(requirements, list) and len(requirements) == 1:
            req = requirements[0]
            
            # Get scaled footprint based on usable area
            footprint = self.parameter_extractor.get_scaled_footprint_for_requirement(
                req, site_contour
            )
            
            # Calculate height from requirement
            height = req.get("n_floors", 1) * req.get("floor_height", 3.5)
            if height <= 0:
                height = 10.0
        else:
            # Fallback for multiple requirements (use old behavior)
            height = self.parameter_extractor.extract_height_from_requirements(requirements)
            footprint = self.parameter_extractor.extract_footprint_from_site_contour(site_contour)
        
        # Generate mesh based on strategy
        if strategy == "simple":
            mesh = self.create_extruded_mesh(footprint, height)
        elif strategy == "setback":
            mesh = self.create_setback_mesh(footprint, height, num_setbacks=3)
        elif strategy == "podium_tower":
            mesh = self.create_podium_tower_mesh(footprint, height)
        elif strategy == "varied":
            mesh = self.create_varied_height_mesh(footprint, height)
        else:
            mesh = self.create_extruded_mesh(footprint, height)

        return mesh
