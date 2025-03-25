import os
import numpy as np
import torch
import folder_paths
from comfy.utils import ProgressBar

from huggingface_hub import snapshot_download

from .pipeline_infu_flux import InfUFluxPipeline


models_dir = folder_paths.models_dir



class InfiniteYou_Load:

    
    def download_models(self):
        print(models_dir)
        snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir= os.path.join(models_dir , 'InfiniteYou'), local_dir_use_symlinks=False)
        try:
            snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir= os.path.join(models_dir , 'FLUX.1-dev'), local_dir_use_symlinks=False)
        except Exception as e:
            print(e)
            print('\nYou are downloading `black-forest-labs/FLUX.1-dev` to `./models/FLUX.1-dev` but failed. '
                'Please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
                'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
                'After that, run the code again.')
            print('\nYou can also download it manually from HuggingFace and put it in `./models/InfiniteYou`, '
                'or you can modify `base_model_path` in `app.py` to specify the correct path.')


    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "model_version": (["sim_stage1","aes_stage2"], {"default": "aes_stage2"}),
            },
            "optional": {
                "base_model_path": ("STRING", { "default": f'{models_dir}\\FLUX.1-dev', "multiline": False }), 
                "realism": ("BOOLEAN", { "default": False }),    
                "anti_blur": ("BOOLEAN", { "default": False }),  
                "need_download": ("BOOLEAN", { "default": False }),              
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",                                
            },
        }
    
    RETURN_TYPES = ("PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "prepare_pipeline"

    CATEGORY = "InfiniteYou"

    def prepare_pipeline(self,base_model_path, model_version, realism, anti_blur, need_download, **kwargs):
        
        if need_download:
            self.download_models()

        pipeline = None

        if pipeline is None or pipeline.model_version != model_version:
            del pipeline

            model_path = f'{models_dir}\\InfiniteYou\\infu_flux_v1.0\\{model_version}'
            print(f'loading model from {model_path}')
            print(f'loading base_model_path {base_model_path}')

            pipeline = InfUFluxPipeline(
                base_model_path=base_model_path,
                infu_model_path=model_path,
                insightface_root_path=f'{models_dir}\\InfiniteYou\\supports\\insightface',
                image_proj_num_tokens=8,
                infu_flux_version='v1.0',
                model_version=model_version,
            )

            if not pipeline.been_loaded:
                return(None,)

        pipeline.pipe.delete_adapters(['realism', 'anti_blur'])
        loras = []
        if realism:
            loras.append([f'{models_dir}\\InfiniteYou\\supports\\optional_loras\\flux_realism_lora.safetensors', 'realism', 1.0])
        if anti_blur:
            loras.append([f'{models_dir}\\InfiniteYou\\supports\\optional_loras\\flux_anti_blur_lora.safetensors', 'anti_blur', 1.0])
        pipeline.load_loras(loras)

        return (pipeline,)
    # @classmethod
    # def VALIDATE_INPUTS(self, Name,Age, **kwargs):
    #     return True


class InfiniteYou_Image:


    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "input_image": ("IMAGE",),                
                "prompt": ("STRING", { "default": "", "multiline": True }),           
            },
            "optional": {
                "control_image": ("IMAGE",),
                "seed": ("INT", {"default": 666666, "min": 1, "max": 9999999999, "step": 1}),
                "width": ("INT", {"default": 1152, }),
                "height": ("INT", {"default": 864, }),
                "num_steps": ("INT", {"default": 30, }),
                "guidance_scale": ("FLOAT", {"default": 3.5, }),
                "conditioning_scale": ("FLOAT", {"default": 1, }),
                "guidance_start": ("FLOAT", {"default": 0, "min": 0, "max": 1,}),
                "guidance_end": ("FLOAT", {"default": 1, "min": 0, "max": 1,}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",                                
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"

    CATEGORY = "InfiniteYou"

    def generate_image(self,pipeline, seed, input_image, control_image, prompt,width, height, num_steps,guidance_scale, conditioning_scale, guidance_start, guidance_end, **kwargs):
        
        if not pipeline:
            return (None, )

        if seed == 0:
            seed = torch.seed() & 0xFFFFFFFF

        image = None
        try:
            image = pipeline(
                id_image=input_image,
                prompt=prompt,
                control_image=control_image,
                seed=seed,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                infusenet_conditioning_scale=conditioning_scale,
                infusenet_guidance_start=guidance_start,
                infusenet_guidance_end=guidance_end,
            )
        except Exception as e:
            print(e)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "InfiniteYou_Load": InfiniteYou_Load,

    "InfiniteYou_Image": InfiniteYou_Image,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteYou_Load": "Load Pipeline",
    "InfiniteYou_Image": "Generate Image",
}
