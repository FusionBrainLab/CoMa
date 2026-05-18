"""Build text prompts for Shap-E from building requirements."""


class PromptBuilder:
    """Converts building requirement dicts into text prompts for text-to-3D generation."""

    FUNCTION_DESCRIPTIONS = {
        "Residential Apartment": "residential apartment building",
        "Performances, Conferences, Ceremonies": "performance hall and conference center",
        "Institutional Accommodation": "institutional accommodation building",
        "Office": "modern office building",
        "Commercial": "commercial building",
        "Mixed Use": "mixed-use building with retail and apartments",
        "Education": "school building",
        "Healthcare": "hospital building",
        "Industrial": "industrial warehouse",
        "Retail": "retail store building",
        "Hotel": "hotel building",
        "Religious": "religious building",
        "Recreation": "recreation center",
        "Parking": "parking structure",
    }

    def build_prompt(self, requirement):
        """
        Create a descriptive text prompt from a building requirement.

        Args:
            requirement: Dict with building_function, n_floors, floor_height,
                         usable_area, building_name, etc.

        Returns:
            Text prompt string suitable for Shap-E.
        """
        building_function = requirement.get("building_function", "")
        n_floors = int(requirement.get("n_floors", 1))
        building_name = requirement.get("building_name", "")

        # Map function to a description Shap-E is more likely to understand
        func_desc = self.FUNCTION_DESCRIPTIONS.get(
            building_function,
            f"{building_function.lower()} building" if building_function else "building",
        )

        # Height category
        if n_floors >= 10:
            height_desc = "tall"
        elif n_floors >= 5:
            height_desc = "mid-rise"
        elif n_floors >= 3:
            height_desc = "low-rise"
        else:
            height_desc = "small"

        prompt = f"a {height_desc} {func_desc}, {n_floors} stories"

        if building_name:
            prompt = f"{building_name}, {prompt}"

        prompt += ", architectural massing, 3D building"
        return prompt

    def build_category_key(self, requirement):
        """
        Return a coarse category key for reference-mesh caching.

        Buildings with the same key share a reference mesh so that we
        don't need to run the diffusion model for every single building.

        Args:
            requirement: Dict with building parameters.

        Returns:
            Tuple (building_function, height_bucket) used as a cache key.
        """
        building_function = requirement.get("building_function", "unknown")
        n_floors = int(requirement.get("n_floors", 1))

        if n_floors >= 10:
            bucket = "high"
        elif n_floors >= 5:
            bucket = "mid"
        elif n_floors >= 3:
            bucket = "low"
        else:
            bucket = "small"

        return (building_function, bucket)
