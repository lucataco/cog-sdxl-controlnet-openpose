# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DiffusionPipeline

CONTROL_NAME = "lllyasviel/ControlNet"
OPENPOSE_NAME = "thibaud/controlnet-openpose-sdxl-1.0"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROL_CACHE = "control-cache"
POSE_CACHE = "pose-cache"
MODEL_CACHE = "model-cache"
REFINER_CACHE = "refiner-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.openpose = OpenposeDetector.from_pretrained(
            CONTROL_NAME,
            cache_dir=CONTROL_CACHE,
        )
        controlnet = ControlNetModel.from_pretrained(
            POSE_CACHE,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self.pipe = pipe.to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            REFINER_CACHE,
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner = refiner.to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input pose image"),
        prompt: str = Input(
            description="Input prompt",
            default="a latina ballerina, romantic sunset, 4k photo",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="low quality, bad quality",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        high_noise_frac: float = Input(
            description="for expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        seed: int = Input(description="Random seed. Set to 0 to randomize the seed", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        refine = "expert_ensemble_refiner"
        refine_steps = None

        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        # Load pose image
        image = Image.open(image).resize((1024, 1024))
        openpose_image = self.openpose(image).resize((1024, 1024))

        sdxl_kwargs = {}
        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac

        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = self.pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            image=openpose_image,
            generator=generator,
        )

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }
            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)
        
        output_path = "./output.png"
        output_image = output.images[0]
        output_image.save(output_path)

        return Path(output_path)