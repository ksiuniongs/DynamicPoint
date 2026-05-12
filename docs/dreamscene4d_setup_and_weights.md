# DreamScene4D Setup And Weights

## Scope

This note summarizes what is required to bring the `DreamScene4D` dynamic branch into the local workspace, with emphasis on:

- environment and install order
- which model weights are actually required
- which models are auto-downloaded from Hugging Face
- which extra repositories are expected by the current code

Repository root:

- `/mnt/d/develop/4D/submodules/dreamscene4d`

## Recommended Environment

The repo README and our local pipeline log both point to a dedicated Python 3.8 + CUDA 11.8 environment:

- Python: `3.8.18`
- PyTorch: `2.2.0`
- torchvision: `0.17.0`
- torchaudio: `2.2.0`
- CUDA runtime/toolkit: `11.8`

References:

- [README.md](/mnt/d/develop/4D/submodules/dreamscene4d/README.md#L13)
- [forest_walk_0806_0810_pipeline.md](/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md#L742)

## Install Order

Recommended setup sequence:

1. Create and activate a dedicated conda env.
2. Install PyTorch with CUDA 11.8.
3. Install the repo-local `diffusers`.
4. Install Python dependencies from `requirements.txt`.
5. Compile local CUDA extensions:
   - `simple-knn`
   - `diff-gaussian-rasterization`
6. Install `nvdiffrast`.
7. Place the required manual checkpoints.
8. Only if needed, install optional external repos such as `WAFT`, `zero123plus`, `sam-3d-objects`, or `Grounded-Segment-Anything`.

Our local log used this shape:

```bash
conda create -n dreamscene4d python=3.8.18
conda activate dreamscene4d

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8.89 cuda-compiler=11.8.0 cuda-libraries-dev=11.8.0

pip install ./diffusers
pip install -r requirements.txt gdown
pip install ./simple-knn
pip install ./diff-gaussian-rasterization
pip install git+https://github.com/NVlabs/nvdiffrast/
```

References:

- [README.md](/mnt/d/develop/4D/submodules/dreamscene4d/README.md#L13)
- [forest_walk_0806_0810_pipeline.md](/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md#L748)

## Minimum Required Weights For The Default Dynamic Pipeline

If you run the normal `main.py -> main_4d.py -> main_4d_compose.py` path with the default configs in this repo, the minimum checkpoint requirement is:

### 1. GMFlow checkpoint

Status:

- required for stage 2 optical flow supervision
- must be downloaded manually

Expected path:

- `/mnt/d/develop/4D/submodules/dreamscene4d/gmflow/pretrained/gmflow_kitti-285701a8.pth`

Why:

- `configs/4d.yaml` sets `gmflow_path` to this file
- `main_4d.py` loads the checkpoint directly with `torch.load`
- the README explicitly says to download it manually

References:

- [README.md](/mnt/d/develop/4D/submodules/dreamscene4d/README.md#L46)
- [4d.yaml](/mnt/d/develop/4D/submodules/dreamscene4d/configs/4d.yaml#L25)
- [main_4d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py#L446)

### 2. Zero123-family guidance weights

Status:

- required by default because both `image.yaml` and `4d.yaml` set `lambda_zero123: 1`
- auto-downloaded from Hugging Face at first use

Default model in current configs:

- `ashawkey/zero123-xl-diffusers`

Optional alternative models already supported by code:

- `ashawkey/stable-zero123-diffusers`
- `sudo-ai/zero123plus-v1.2`

Why:

- `main.py` and `main_4d.py` enable Zero123 guidance whenever `lambda_zero123 > 0`
- `Zero123Pipeline.from_pretrained(...)` and `DiffusionPipeline.from_pretrained(...)` are used directly

References:

- [image.yaml](/mnt/d/develop/4D/submodules/dreamscene4d/configs/image.yaml#L23)
- [4d.yaml](/mnt/d/develop/4D/submodules/dreamscene4d/configs/4d.yaml#L31)
- [main.py](/mnt/d/develop/4D/submodules/dreamscene4d/main.py#L174)
- [main_4d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py#L263)
- [zero123_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/zero123_utils.py#L27)
- [zero123plus_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/zero123plus_utils.py#L27)

### 3. Depth Anything

Status:

- not required in the default configs because `depth_loss: False`
- auto-downloaded from Hugging Face if enabled

Models used in code:

- `LiheYoung/depth-anything-large-hf`
- `LiheYoung/depth-anything-small-hf`

Why:

- stage 1 and stage 2 call `transformers.pipeline("depth-estimation", ...)` only when depth loss or compose depth estimation is enabled

References:

- [main.py](/mnt/d/develop/4D/submodules/dreamscene4d/main.py#L553)
- [main_4d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py#L927)
- [main_4d_compose.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_4d_compose.py#L429)

## What Is Mandatory Vs Optional

### Mandatory for the default repo config

- `gmflow/pretrained/gmflow_kitti-285701a8.pth`
- network access or pre-cached Hugging Face weights for Zero123

### Optional if you change config or workflow

- WAFT checkpoint
- Zero123++ external pipeline repo
- Stable Diffusion text guidance weights
- Stable Video Diffusion weights
- MVDream model weights
- trackseg segmentation weights
- SAM3D checkpoints

## Optional Weights And External Repos

### WAFT

Status:

- optional replacement for GMFlow
- only needed if `flow_backend=waft`

Expected paths from config:

- `/mnt/d/develop/4D/submodules/WAFT/config/a1/tar-c-t.json`
- `/mnt/d/develop/4D/submodules/WAFT/ckpts/a1/adaptation.pth`

Why:

- `main_4d.py` switches between GMFlow and WAFT
- `utils/flow_utils.py` hard-errors if WAFT files are missing

References:

- [4d.yaml](/mnt/d/develop/4D/submodules/dreamscene4d/configs/4d.yaml#L25)
- [main_4d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py#L433)

### Zero123++

Status:

- optional alternative to vanilla Zero123
- needs both model weights from Hugging Face and a local helper repo

Expected local repo path:

- `/mnt/d/develop/4D/submodules/zero123plus`

Expected Hugging Face model:

- `sudo-ai/zero123plus-v1.2`

Why:

- the code uses `custom_pipeline=<zero123plus_root>/diffusers-support`
- so Hugging Face weights alone are not enough; the local `zero123plus` repo is also required

References:

- [4d.yaml](/mnt/d/develop/4D/submodules/dreamscene4d/configs/4d.yaml#L37)
- [zero123plus_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/zero123plus_utils.py#L33)

### Stable Diffusion text guidance

Status:

- optional
- only needed if `lambda_sd > 0`
- auto-downloaded from Hugging Face

Supported default model keys:

- `stabilityai/stable-diffusion-2-1-base`
- `stabilityai/stable-diffusion-2-base`
- `runwayml/stable-diffusion-v1-5`

Reference:

- [sd_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/sd_utils.py#L27)

### Stable Video Diffusion

Status:

- optional
- only needed if `lambda_svd > 0`
- auto-downloaded from Hugging Face

Model:

- `stabilityai/stable-video-diffusion-img2vid`

Reference:

- [svd_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/svd_utils.py#L15)

### MVDream

Status:

- optional
- only needed if `mvdream: True`
- may need an external `mvdream` Python package and possibly a checkpoint depending on that package setup

Code behavior:

- `build_model(model_name='sd-v2.1-base-4view', ckpt_path=None)`

Reference:

- [mvdream_utils.py](/mnt/d/develop/4D/submodules/dreamscene4d/guidance/mvdream_utils.py#L12)

### trackseg / mask extraction

Status:

- optional
- only needed if you want DreamScene4D to start from raw video instead of pre-made masks

Install notes in repo README:

- install `trackseg`
- run `bash scripts/download_models.sh`
- install the forked `Grounded-Segment-Anything`

The bundled model download script fetches:

- `DEVA-propagation.pth`
- `groundingdino_swint_ogc.pth`
- `sam_hq_vit_h.pth`
- `GroundingDINO_SwinT_OGC.py`

There are also optional paths in config for:

- `sam_vit_h_4b8939.pth`
- `sam_hq_vit_tiny.pth`
- `mobile_sam.pt`

References:

- [README.md](/mnt/d/develop/4D/submodules/dreamscene4d/README.md#L236)
- [download_models.sh](/mnt/d/develop/4D/submodules/dreamscene4d/trackseg/scripts/download_models.sh#L1)
- [ext_eval_args.py](/mnt/d/develop/4D/submodules/dreamscene4d/trackseg/deva/ext/ext_eval_args.py#L5)

### SAM3D initialization path

Status:

- optional
- only needed if using `main_sam3d.py` or `main_factory.py`

Expected external repo:

- `/mnt/d/develop/4D/submodules/sam-3d-objects`

Expected checkpoint/config location:

- `/mnt/d/develop/4D/submodules/sam-3d-objects/checkpoints/hf/pipeline.yaml`

Important:

- this is not bundled inside `dreamscene4d`
- the code explicitly errors and tells you to download checkpoints from the official Hugging Face repo first

References:

- [main_sam3d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_sam3d.py#L24)
- [main_sam3d.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_sam3d.py#L264)
- [main_factory.py](/mnt/d/develop/4D/submodules/dreamscene4d/main_factory.py#L38)

## Practical Pull List

If the goal is only to run the default dynamic object pipeline, pull these first:

1. `DreamScene4D` repo itself
2. local `diffusers` install from the repo
3. compiled CUDA extensions
4. `gmflow_kitti-285701a8.pth`
5. Hugging Face cache access for `ashawkey/zero123-xl-diffusers`

If the goal is to keep the codebase ready for the optional paths too, add:

1. `/mnt/d/develop/4D/submodules/zero123plus`
2. `/mnt/d/develop/4D/submodules/WAFT`
3. `/mnt/d/develop/4D/submodules/sam-3d-objects`
4. `trackseg` model files under `trackseg/saves`

## Local Status Snapshot

From our current workspace notes:

- GMFlow checkpoint was already prepared once in the dedicated env workflow
- the repo needed a small CUDA 11.8 compatibility patch for local extension builds

References:

- [forest_walk_0806_0810_pipeline.md](/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md#L742)
- [dreamscene4d_cuda11_compat.patch](/mnt/d/develop/master_thesis/DynamicPoint/patches/dreamscene4d_cuda11_compat.patch)

## Short Conclusion

For the default `DreamScene4D` dynamic branch, the only clearly manual hard requirement is:

- `gmflow/pretrained/gmflow_kitti-285701a8.pth`

Everything else on the core path is either:

- auto-fetched from Hugging Face at runtime, or
- only needed if you deliberately enable optional branches such as `WAFT`, `Zero123++`, `trackseg`, or `SAM3D`.
