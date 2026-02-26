"""
Realistic Vision v5.1 — Image-to-Image Pipeline Manager with IP-Adapter Face.
Handles lazy-loading and VRAM error handling.
Optimized for low-VRAM GPUs (e.g. RTX 3050 4GB).

Model: SG161222/Realistic_Vision_V5.1_noVAE
VAE:   stabilityai/sd-vae-ft-mse (better color fidelity)
Face:  h94/IP-Adapter (ip-adapter-plus-face_sd15) — preserves facial identity

Modes:
  - Text-to-Image: generate from prompt only
  - Image-to-Image + Face Preservation: generate from prompt + reference image
    Uses IP-Adapter Face to inject facial features into the generation.

Quality enhancements:
  - Auto quality-boost prompt tags for Realistic Vision
  - Enhanced negative prompts
  - Optional Hi-Res upscale via img2img refinement
"""

import gc
import time
import torch
from PIL import Image as PILImage
from transformers import CLIPVisionModelWithProjection
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

# ---------------------------------------------------------------------------
# Global performance flags
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True          # faster convolutions
torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere+
torch.backends.cudnn.allow_tf32 = True

BASE_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sd15.bin"

SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
}

# ---------------------------------------------------------------------------
# Quality boost tags — tuned for Realistic Vision v5.1
# ---------------------------------------------------------------------------
QUALITY_BOOST_PREFIX = (
    "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, "
    "film grain, Fujifilm XT3, "
)

# Enhanced negative prompt — combined with user's negative prompt
QUALITY_NEGATIVE_SUFFIX = (
    ", (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
    "sketch, cartoon, drawing, anime:1.4), text, close up, cropped, "
    "out of frame, worst quality, low quality, jpeg artifacts, ugly, "
    "duplicate, morbid, mutilated, extra fingers, mutated hands, "
    "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, "
    "dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, "
    "disfigured, gross proportions, malformed limbs, missing arms, "
    "missing legs, extra arms, extra legs, fused fingers, too many fingers, "
    "long neck"
)


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str):
    """Return optimal dtype for device."""
    if device == "cuda":
        return torch.float16
    return torch.float32


class PipelineManager:
    """Manages SD pipelines with IP-Adapter Face for identity preservation."""

    # VRAM threshold (GB) — below this we enable aggressive memory saving
    LOW_VRAM_THRESHOLD = 6.0

    def __init__(self):
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self._vae = None
        self._pipe_sd = None
        self._pipe_img2img = None
        self._image_encoder = None
        # We'll set this to True once load_ip_adapter is called
        self._ip_adapter_loaded = False
        self._low_vram = self._detect_low_vram()
        if self._low_vram:
            print(f"[Pipeline] Low-VRAM mode ENABLED (< {self.LOW_VRAM_THRESHOLD} GB)")

    @staticmethod
    def _detect_low_vram() -> bool:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return vram_gb < PipelineManager.LOW_VRAM_THRESHOLD
        return False

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _apply_memory_opts(self, pipe):
        """Apply memory-saving optimizations to a pipeline."""
        pipe.enable_attention_slicing("auto")
        # VAE optimizations — drastically reduce VRAM spikes
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        if self.device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[Pipeline] xformers enabled ✓")
            except Exception:
                pass
        return pipe

    def _load_vae(self):
        """Load the better VAE (sd-vae-ft-mse) for improved color fidelity."""
        if not hasattr(self, '_vae') or self._vae is None:
            print(f"[Pipeline] Loading Better VAE ({VAE_MODEL_ID})...")
            self._vae = AutoencoderKL.from_pretrained(
                VAE_MODEL_ID,
                torch_dtype=self.dtype,
            )
        return self._vae

    def _load_image_encoder(self):
        """Load the CLIP image encoder needed for IP-Adapter."""
        if self._image_encoder is None:
            print("[Pipeline] Loading CLIP image encoder for IP-Adapter...")
            self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                IP_ADAPTER_REPO,
                subfolder="models/image_encoder",
                torch_dtype=self.dtype,
            )
        return self._image_encoder

    def _get_sd_pipeline(self):
        """Load base SD pipeline (text-to-image)."""
        if self._pipe_sd is None:
            t0 = time.time()
            vae = self._load_vae()
            image_encoder = self._load_image_encoder()
            print(f"[Pipeline] Loading Realistic Vision v5.1 on {self.device} ({self.dtype})...")
            self._pipe_sd = StableDiffusionPipeline.from_pretrained(
                BASE_MODEL_ID,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
            self._apply_memory_opts(self._pipe_sd)

            # Load IP-Adapter Face weights
            print("[Pipeline] Loading IP-Adapter Face (plus-face)...")
            self._pipe_sd.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder="models",
                weight_name=IP_ADAPTER_WEIGHT,
            )
            self._ip_adapter_loaded = True
            # Default scale 0 = disabled (only active when face image provided)
            self._pipe_sd.set_ip_adapter_scale(0.0)

            print(f"[Pipeline] Realistic Vision + IP-Adapter loaded in {time.time()-t0:.1f}s")
        return self._pipe_sd

    def _get_img2img_pipeline(self):
        """Load img2img pipeline for image-to-image generation."""
        if self._pipe_img2img is None:
            t0 = time.time()
            vae = self._load_vae()
            image_encoder = self._load_image_encoder()
            print(f"[Pipeline] Loading img2img pipeline...")
            self._pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                BASE_MODEL_ID,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
            self._apply_memory_opts(self._pipe_img2img)

            # Load IP-Adapter Face weights
            print("[Pipeline] Loading IP-Adapter Face for img2img...")
            self._pipe_img2img.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder="models",
                weight_name=IP_ADAPTER_WEIGHT,
            )
            self._ip_adapter_loaded = True
            self._pipe_img2img.set_ip_adapter_scale(0.0)

            print(f"[Pipeline] img2img + IP-Adapter loaded in {time.time()-t0:.1f}s")
        return self._pipe_img2img

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _set_scheduler(pipe, sampler_name: str):
        """Switch the scheduler on the pipeline."""
        sched_cls = SCHEDULERS.get(sampler_name, EulerDiscreteScheduler)
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)

    # ------------------------------------------------------------------
    # Face cropping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_face_region(image: PILImage.Image) -> PILImage.Image:
        """
        Crop and resize the face area from the reference image.
        Uses a simple center-crop heuristic (top 60% of image, centered).
        This works well for selfies and portrait photos.
        """
        w, h = image.size
        # For selfies/portraits: crop upper-center region where face typically is
        face_h = int(h * 0.65)  # top 65%
        face_w = int(w * 0.70)  # center 70%
        left = (w - face_w) // 2
        top = 0
        right = left + face_w
        bottom = face_h

        face_crop = image.crop((left, top, right, bottom))
        # Resize to 224x224 as expected by CLIP image encoder
        face_crop = face_crop.resize((224, 224), PILImage.LANCZOS)
        return face_crop

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_prompt(prompt: str, quality_boost: bool = True) -> str:
        """Prepend quality tags to improve SD 1.5 output quality."""
        if not quality_boost:
            return prompt
        # Avoid double-adding if user already typed quality tags
        lower = prompt.lower()
        if "masterpiece" in lower or "best quality" in lower:
            return prompt
        return QUALITY_BOOST_PREFIX + prompt

    @staticmethod
    def _enhance_negative(negative_prompt: str) -> str:
        """Append quality-related negative tokens."""
        if not negative_prompt:
            return QUALITY_NEGATIVE_SUFFIX.lstrip(", ")
        # Avoid duplicating if already present
        if "worst quality" in negative_prompt.lower():
            return negative_prompt
        return negative_prompt + QUALITY_NEGATIVE_SUFFIX

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1,
        sampler: str = "euler",
        # Image-to-Image params
        reference_image=None,
        strength: float = 0.55,
        # IP-Adapter Face params
        face_strength: float = 0.6,
        # Quality params
        quality_boost: bool = True,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_strength: float = 0.45,
    ):
        """
        Run the diffusion pipeline and return a PIL Image.

        If reference_image is provided, runs img2img mode with IP-Adapter Face
        to preserve facial identity from the reference image.

        Params:
          reference_image – PIL Image to use as starting point (img2img)
          strength        – denoising strength (0.0 = no change, 1.0 = fully new)
                            Lower = closer to original, Higher = more creative
          face_strength   – IP-Adapter face influence (0.0 = ignore face, 1.0 = max)
                            Controls how strongly the face from reference is preserved
          quality_boost   – auto-prepend quality tags to prompt
          hires_fix       – upscale image via img2img refinement pass
          hires_scale     – how much to upscale (1.5x default)
          hires_strength  – denoising strength for refine pass
        """
        # For img2img: disable quality boost by default to let user prompt
        # take full effect (quality prefix can override clothing/scene changes)
        if reference_image is not None and quality_boost:
            # Auto-disable quality boost for img2img to improve prompt adherence
            print("[Pipeline] img2img detected: quality boost auto-disabled for better prompt adherence")
            quality_boost = False

        # Enhance prompts for quality
        enhanced_prompt = self._enhance_prompt(prompt, quality_boost)
        enhanced_neg = self._enhance_negative(negative_prompt)

        print(f"[Pipeline] Prompt (enhanced): {enhanced_prompt[:120]}...")
        if reference_image is not None:
            print(f"[Pipeline] Mode: Image-to-Image + IP-Adapter Face")
            print(f"[Pipeline]   Denoising strength={strength}, Face strength={face_strength}")
        else:
            print(f"[Pipeline] Mode: Text-to-Image")

        # Determine seed
        if seed < 0:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            # Clear VRAM before generation to maximize available memory
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

            t0 = time.time()

            # Use inference_mode for faster execution (no grad tracking)
            with torch.inference_mode():
                if reference_image is not None:
                    # ── Image-to-Image + IP-Adapter Face mode ──

                    # Crop face from reference image for IP-Adapter
                    face_image = self._crop_face_region(reference_image)
                    print(f"[Pipeline] Face region cropped for IP-Adapter (224x224)")

                    pipe = self._get_img2img_pipeline()
                    self._set_scheduler(pipe, sampler)

                    # Enable IP-Adapter with face strength
                    pipe.set_ip_adapter_scale(face_strength)

                    result = pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_neg,
                        image=reference_image,
                        ip_adapter_image=face_image,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )

                    # Reset IP-Adapter scale after generation
                    pipe.set_ip_adapter_scale(0.0)
                else:
                    # ── Text-to-Image mode ──
                    pipe = self._get_sd_pipeline()
                    self._set_scheduler(pipe, sampler)

                    # IMPORTANT: Disable IP-Adapter for Text-to-Image
                    pipe.set_ip_adapter_scale(0.0)

                    result = pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_neg,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        # Pass a dummy black image to avoid NoneType iteration in some diffusers versions
                        ip_adapter_image=PILImage.new("RGB", (224, 224), (0, 0, 0)) if self._ip_adapter_loaded else None,
                    )

            image = result.images[0]

            # ── Hi-Res Fix: upscale + img2img refinement pass ──
            if hires_fix and not self._low_vram:
                print(f"[Pipeline] Hi-Res Fix: upscaling {hires_scale}x with strength {hires_strength}...")
                new_w = int(image.width * hires_scale)
                new_h = int(image.height * hires_scale)
                # Round to nearest multiple of 8
                new_w = max(64, (new_w // 8) * 8)
                new_h = max(64, (new_h // 8) * 8)

                # Upscale the image first using Lanczos
                image_upscaled = image.resize((new_w, new_h), PILImage.LANCZOS)

                # Run img2img refinement pass
                if self.device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

                generator2 = torch.Generator(device=self.device).manual_seed(seed)
                pipe_i2i = self._get_img2img_pipeline()
                self._set_scheduler(pipe_i2i, sampler)

                # For hi-res fix, also pass face image if available
                hires_kwargs = {}
                if reference_image is not None:
                    face_image = self._crop_face_region(reference_image)
                    pipe_i2i.set_ip_adapter_scale(face_strength * 0.5)  # lower for refinement
                    hires_kwargs["ip_adapter_image"] = face_image

                with torch.inference_mode():
                    refine_result = pipe_i2i(
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_neg,
                        image=image_upscaled,
                        strength=hires_strength,
                        num_inference_steps=max(10, num_inference_steps // 2),
                        guidance_scale=guidance_scale,
                        generator=generator2,
                        ip_adapter_image=hires_kwargs.get("ip_adapter_image", PILImage.new("RGB", (224, 224), (0, 0, 0)) if self._ip_adapter_loaded else None),
                    )

                pipe_i2i.set_ip_adapter_scale(0.0)
                image = refine_result.images[0]
                print(f"[Pipeline] Hi-Res Fix done → {new_w}x{new_h}")
            elif hires_fix and self._low_vram:
                # Low VRAM: just do Lanczos upscale without img2img
                print(f"[Pipeline] Hi-Res Fix (low-VRAM mode): Lanczos upscale only...")
                new_w = int(image.width * hires_scale)
                new_h = int(image.height * hires_scale)
                new_w = max(64, (new_w // 8) * 8)
                new_h = max(64, (new_h // 8) * 8)
                image = image.resize((new_w, new_h), PILImage.LANCZOS)

            elapsed = time.time() - t0
            print(f"[Pipeline] Generation done in {elapsed:.1f}s ({num_inference_steps} steps)")

            return image, seed

        except torch.cuda.OutOfMemoryError:
            # Clear VRAM and report
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "VRAM tidak cukup! Coba kurangi resolusi atau sampling steps, atau matikan Hi-Res Fix."
            )
        except Exception as e:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(f"Error saat generate: {str(e)}")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current device / model status."""
        gpu_name = None
        vram = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        return {
            "device": self.device,
            "gpu": gpu_name,
            "vram": vram,
            "dtype": str(self.dtype),
        }
