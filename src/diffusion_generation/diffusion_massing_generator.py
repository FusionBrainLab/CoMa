"""Generate building massings using the Shap-E text-to-3D diffusion model."""
import os
import tempfile

import numpy as np
import torch
import trimesh

from .prompt_builder import PromptBuilder
from .mesh_postprocessor import MeshPostprocessor
from ..procedural_generation.parameter_extractor import ParameterExtractor


class DiffusionMassingGenerator:
    """
    Generate building massings by:
    1. Building a text prompt from the requirement.
    2. Running Shap-E to produce a 3D mesh.
    3. Extracting a height-profile (how cross-section varies with height).
    4. Applying that profile to a correctly-scaled footprint derived from the
       site contour and the requirement's usable-area / n-floors.

    Two generation modes are supported:
    * ``"per_building"`` -- generate a unique mesh for every building (slow).
    * ``"reference"``    -- generate one reference mesh per (function, height)
      category and re-use its profile for every building in that category (fast).
    """

    def __init__(
        self,
        model_name="openai/shap-e",
        device=None,
        num_inference_steps=64,
        guidance_scale=15.0,
        mode="reference",
        seed=42,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.mode = mode
        self.seed = seed

        self.prompt_builder = PromptBuilder()
        self.postprocessor = MeshPostprocessor()
        self.parameter_extractor = ParameterExtractor()

        # Cache: category_key -> height_profile (list of floats)
        self._profile_cache = {}

        # Shap-E pipeline (loaded lazily)
        self._pipe = None
        self._pipe_loaded = False

    # ------------------------------------------------------------------
    # Shap-E pipeline management
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        """Load the Shap-E pipeline from HuggingFace (downloads weights on first call)."""
        if self._pipe_loaded:
            return

        try:
            from diffusers import ShapEPipeline
            CACHE_DIR = '/home/jovyan/SR008.fs2/transformers_cache'

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipe = ShapEPipeline.from_pretrained(
                self.model_name, torch_dtype=dtype, cache_dir=CACHE_DIR
            )
            self._pipe = self._pipe.to(self.device)
            print(f"Shap-E pipeline loaded on {self.device}")
        except Exception as exc:
            print(f"WARNING: Could not load Shap-E pipeline: {exc}")
            print("Falling back to noise-based profile generation.")
            self._pipe = None

        self._pipe_loaded = True

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def _generate_mesh_shap_e(self, prompt, seed=None):
        """
        Run Shap-E and return a trimesh.Trimesh.

        The pipeline outputs latent 3-D representations that are decoded to
        a mesh and exported via a temporary PLY file.
        """
        self._load_pipeline()
        if self._pipe is None:
            return None

        gen_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(gen_seed)

        output = self._pipe(
            prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            frame_size=256,
            output_type="mesh",
        )

        mesh_output = output.images[0]

        # --- try direct vertex / face access ---
        verts = getattr(mesh_output, "verts", None)
        faces = getattr(mesh_output, "faces", None)
        if verts is not None and faces is not None:
            verts_np = verts.cpu().numpy() if torch.is_tensor(verts) else np.asarray(verts)
            faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
            if len(verts_np) > 0 and len(faces_np) > 0:
                return trimesh.Trimesh(vertices=verts_np, faces=faces_np)

        # --- fallback: export to PLY and reload ---
        tmp_path = None
        try:
            from diffusers.utils import export_to_ply

            fd, tmp_path = tempfile.mkstemp(suffix=".ply")
            os.close(fd)
            export_to_ply(mesh_output, tmp_path)
            return trimesh.load(tmp_path)
        except Exception as exc:
            print(f"  PLY export failed: {exc}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Noise-based fallback profile
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_noise_profile(n_samples, seed=42):
        """
        Create a plausible building height-profile from noise when Shap-E
        is unavailable.

        The profile is monotonically non-increasing (wider at bottom) with
        stochastic variation, mimicking a reverse-diffusion sampling process
        over the parameter space.
        """
        rng = np.random.RandomState(seed)
        raw = rng.rand(n_samples)
        # Apply a decaying envelope so upper floors are narrower
        envelope = np.linspace(1.0, 0.4, n_samples)
        profile = np.clip(raw * 0.3 + envelope * 0.7, 0.1, 1.0)
        # Ensure monotonically non-increasing
        for i in range(1, len(profile)):
            profile[i] = min(profile[i], profile[i - 1])
        return profile.tolist()

    # ------------------------------------------------------------------
    # Profile extraction (with caching)
    # ------------------------------------------------------------------

    def _get_profile(self, requirement, profile_samples=20):
        """
        Return a height-profile list for the given requirement.

        In ``reference`` mode the profile is cached by category key so
        that the expensive diffusion step runs at most once per category.
        """
        category = self.prompt_builder.build_category_key(requirement)

        if self.mode == "reference" and category in self._profile_cache:
            return self._profile_cache[category]

        prompt = self.prompt_builder.build_prompt(requirement)
        mesh = self._generate_mesh_shap_e(prompt)

        if mesh is not None and len(mesh.vertices) > 0:
            profile = self.postprocessor.extract_height_profile(mesh, n_samples=profile_samples)
        else:
            # Noise-based fallback
            cat_seed = hash(category) % (2**31)
            profile = self._generate_noise_profile(profile_samples, seed=cat_seed)

        if self.mode == "reference":
            self._profile_cache[category] = profile

        return profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_massing(self, requirement, site_contour):
        """
        Generate a single building massing from a requirement and site contour.

        Args:
            requirement: Dict with id, n_floors, floor_height,
                usable_area, building_function, etc.
            site_contour: List of polygons (list of list of [x, y]).

        Returns:
            Dict {"id": ..., "massing": [...]} in the dataset format.
        """
        building_id = requirement.get("id", "0")
        n_floors = int(requirement.get("n_floors", 1))
        floor_height = float(requirement.get("floor_height", 3.5))

        if n_floors <= 0:
            n_floors = 1
        if floor_height <= 0:
            floor_height = 3.5

        # Get the correctly-sized footprint from the parametric extractor
        footprint = self.parameter_extractor.get_scaled_footprint_for_requirement(
            requirement, site_contour
        )

        # Obtain the height-profile from the diffusion model (or fallback)
        profile = self._get_profile(requirement)

        # Resample profile to one value per floor
        floor_scales = self.postprocessor.sample_profile_for_floors(profile, n_floors)

        # Assemble the massing dict
        massing = self.postprocessor.build_massing(
            footprint, n_floors, floor_height, floor_scales, building_id
        )

        return massing
