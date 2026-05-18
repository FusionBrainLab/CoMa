"""Extract and calculate parameters from requirements and site contour."""
from .mesh_utils import scale_polygon_to_area, calculate_polygon_area


class ParameterExtractor:
    """Extract building parameters from requirements and site data."""
    
    def extract_height_from_requirements(self, requirements):
        """
        Calculate total height as n_floors × floor_height from requirements.
        
        Args:
            requirements: List of requirement dicts with n_floors and floor_height
        
        Returns:
            Total height in meters (uses max if multiple requirements)
        """
        total_height = 0

        for req in requirements:
            n_floors = req.get("n_floors", 0)
            floor_height = req.get("floor_height", 3.5)
            building_height = n_floors * floor_height
            total_height = max(total_height, building_height)

        if total_height <= 0:
            total_height = 10.0  # Fallback default

        return total_height

    def extract_footprint_from_site_contour(self, site_contour):
        """
        Extract first polygon from site contour as building footprint.
        
        Args:
            site_contour: Site contour polygon data
        
        Returns:
            List of [x, y] coordinates representing the footprint
        """
        if isinstance(site_contour, list) and len(site_contour) > 0:
            if isinstance(site_contour[0], list) and len(site_contour[0]) > 0:
                footprint = site_contour[0]
            else:
                footprint = site_contour
        else:
            footprint = [[0, 0], [10, 0], [10, 10], [0, 10]]  # Fallback square

        return footprint
    
    def calculate_required_footprint_area(self, requirement):
        """
        Calculate required footprint area based on usable area and floor count.
        
        Formula: footprint_area = usable_area / n_floors
        
        Args:
            requirement: Dict with usable_area and n_floors
        
        Returns:
            Required footprint area in square meters
        """
        usable_area = requirement.get("usable_area", 0)
        n_floors = requirement.get("n_floors", 1)
        
        if n_floors <= 0:
            n_floors = 1
        
        if usable_area <= 0:
            # Fallback: use a default area
            return 100.0
        
        footprint_area = usable_area / n_floors
        return footprint_area
    
    def get_scaled_footprint_for_requirement(self, requirement, site_contour):
        """
        Get a footprint scaled to match the required usable area.
        
        This ensures the building has the correct height and usable area
        by adjusting the footprint size.
        
        Args:
            requirement: Dict with usable_area and n_floors
            site_contour: Site contour polygon (will be scaled down if needed)
        
        Returns:
            Scaled footprint polygon as list of [x, y] coordinates
        """
        # Get base footprint from site contour
        base_footprint = self.extract_footprint_from_site_contour(site_contour)
        
        # Calculate required footprint area
        required_area = self.calculate_required_footprint_area(requirement)
        
        # Scale the footprint to match required area
        scaled_footprint = scale_polygon_to_area(base_footprint, required_area)
        
        return scaled_footprint
