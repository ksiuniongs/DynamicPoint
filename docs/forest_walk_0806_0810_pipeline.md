# Forest Walk 0806-0810 Pipeline

## Scope

This file records the concrete preprocessing and reconstruction steps executed for:

- Source video: `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/YTDown.com_YouTube_360-VR-Forest-Walk-8K-Virtual-Relaxation_Media_G_gmoSejUxU_001_1080s.mp4`
- Working root: `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810`

The goal is to keep an end-to-end runnable record: which script was used, where it lives, how it was run, and what it produced.

## Current Assets

- SCGS-style view extractor:
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/extract_scgs_views.py`
- COLMAP stereo flattening helper:
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_colmap_stereo_input.py`
- SAM3 tracking/segmentation entrypoint:
  - `/mnt/d/develop/4D/run_sam3_trackseg.py`
- Pipeline log appender:
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/append_pipeline_log.py`

## Executed Steps

### 1. Inspect `data/Dataset`

Purpose:
- Understand scene layout and target image sizing convention.

Findings:
- Scene folders: `Kayak`, `Lake`, `OpenSea`, `Outback`, `Picnic`, `ShortRide`, `Tree`, `Tunnel`
- Typical image sizes:
  - 4:3-like: `1439x1079`, `1437x1076`, `1432x1076`
  - 16:9-like: `1921x1078`, `1919x1079`, `1919x1078`

Decision:
- Use `1439x1079` as the target 4:3 size for this sequence.

### 2. Extract SCGS-style views from the 360 video

Purpose:
- Follow the paper-style fixed virtual view setup instead of mixing views or using arbitrary projection combinations.

Script:
- `/mnt/d/develop/master_thesis/DynamicPoint/scripts/extract_scgs_views.py`

Key behavior:
- Extract frames from the 360 source video.
- Render 7 fixed views only:
  - `(yaw, pitch) = (0, 0)`
  - `(-30, 0)`, `(30, 0)`
  - `(-60, 0)`, `(60, 0)`
  - `(0, -10)`, `(0, 10)`

Command:

```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/extract_scgs_views.py \
  --video /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/YTDown.com_YouTube_360-VR-Forest-Walk-8K-Virtual-Relaxation_Media_G_gmoSejUxU_001_1080s.mp4 \
  --start 00:08:06 \
  --duration 4 \
  --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810 \
  --fps 30 \
  --size 1439x1079 \
  --fov 90 \
  --views_dir_name views_scgs_4x3
```

Outputs:
- Frames:
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/frames`
- Views:
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3`

Counts:
- `120` raw frames
- `7 x 120` projected view images

### 3. Select the left/right stereo pair

Purpose:
- Reduce the 7-view set to a stereo pair for sparse reconstruction.

Mapping:
- `yaw_m030_pitch_000` -> `left`
- `yaw_030_pitch_000` -> `right`

Selection rule:
- Keep only the last `20` frames.
- Renumber to `frame_000000.jpg` ... `frame_000019.jpg`

Current directories:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left`
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right`

### 4. Build COLMAP input from `left/right`

Purpose:
- Flatten the stereo pair into a single image directory that COLMAP can ingest.

Script:
- `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_colmap_stereo_input.py`

Command:

```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_colmap_stereo_input.py \
  --left /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left \
  --right /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right \
  --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair
```

Outputs:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/images`

Flattened filenames:
- `000000_left.jpg`, `000000_right.jpg`, ..., `000019_left.jpg`, `000019_right.jpg`

### 5. Run COLMAP sparse reconstruction

Commands:

```bash
colmap feature_extractor \
  --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/database.db \
  --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0
```

```bash
colmap exhaustive_matcher \
  --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/database.db \
  --SiftMatching.use_gpu 0
```

```bash
colmap mapper \
  --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/database.db \
  --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/images \
  --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/sparse
```

Main output:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/sparse/0`

Measured result:
- `40` registered images
- `5718` sparse points
- mean reprojection error about `1.47 px`

### 6. Convert the COLMAP scene into 3DGS-compatible layout

Script:
- `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/convert.py`

Command:

```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/convert.py \
  -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair \
  --skip_matching \
  --no_gpu
```

Outputs:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/input`
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/distorted`
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/stereo`

### 7. Export COLMAP sparse point cloud to PLY

Command:

```bash
colmap model_converter \
  --input_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/sparse/0 \
  --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/ply/points3D.ply \
  --output_type PLY
```

Outputs:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair/ply/points3D.ply`
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/supersplat scenes/forest_walk_0806_0810_sparse.ply`

Important:
- This is a COLMAP sparse point cloud PLY, not a Gaussian Splatting `point_cloud.ply`.
- It does not contain GS fields such as `scale_*`, `rot_*`, `f_dc_*`, `opacity`.

### 8. Generate person masks with SAM3

Environment:
- `/root/miniconda3/envs/sam3`

Script:
- `/mnt/d/develop/4D/run_sam3_trackseg.py`

Commands:

```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/4D/run_sam3_trackseg.py \
  --frames /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right \
  --out /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person \
  --text "a person"
```

```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/4D/run_sam3_trackseg.py \
  --frames /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left \
  --out /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2 \
  --text "a person"
```

Outputs:
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person`
- `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2`

Notes:
- `left_masks_person_v2` is the valid left result after rerunning serially.
- The older `left_masks_person` directory should not be treated as authoritative.

## Future Logging Rule

For every subsequent operation in this pipeline, append a new section at the end of this file with:

1. Date/time
2. Goal
3. Scripts used and absolute paths
4. Exact command(s)
5. Input path(s)
6. Output path(s)
7. Key observations or caveats

The preferred helper is:

```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/append_pipeline_log.py \
  --doc /mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md \
  --title "Short title" \
  --goal "What this step is doing" \
  --script "/abs/path/to/script.py" \
  --command "exact command line" \
  --inputs "/abs/in1" \
  --inputs "/abs/in2" \
  --outputs "/abs/out1" \
  --notes "Important note"
```

### LaMa Setup

- Time: `2026-03-17 06:32:21`
- Goal: Install LaMa, add compatibility patches for inference, and prepare paired image/mask input for inpainting.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/4D/submodules/lama/saicinpainting/training/data/aug.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py`
- Command(s):
```bash
/root/miniconda3/envs/sam3/bin/python -m pip install hydra-core omegaconf albumentations kornia pytorch-lightning scikit-image easydict webdataset scikit-learn pandas matplotlib joblib packaging tabulate
```
```bash
PYTHONPATH=/mnt/d/develop/4D/submodules/lama /root/miniconda3/envs/sam3/bin/python /mnt/d/develop/4D/submodules/lama/bin/predict.py --help
```
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py --images /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left --masks /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2 --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input --prefix left
```
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py --images /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --masks /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input --prefix right
```
- Input(s):
  - `/root/miniconda3/envs/sam3`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person`
- Output(s):
  - `/mnt/d/develop/4D/submodules/lama`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input`
- Note(s):
  - Used the valid left mask directory left_masks_person_v2.
  - Patched aug.py for new albumentations import behavior.
  - Patched predict.py to honor predict_config.device instead of forcing CPU.

### LaMa Inpainting Output

- Time: `2026-03-17 06:36:04`
- Goal: Run LaMa inpainting on left/right person masks and assemble a new COLMAP-style input directory.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/4D/submodules/lama/saicinpainting/training/trainers/__init__.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/lama && PYTHONPATH=/mnt/d/develop/4D/submodules/lama /root/miniconda3/envs/sam3/bin/python bin/predict.py model.path=/mnt/d/develop/4D/submodules/lama/big-lama indir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input outdir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output dataset.img_suffix=.jpg device=cuda:0
```
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py --lama_outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama/images
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input`
  - `/mnt/d/develop/4D/submodules/lama/big-lama`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama/images`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama/input`
- Note(s):
  - LaMa output filenames follow the mask names, so assemble_lama_inpainted_input.py strips _mask001 and restores frame-first COLMAP naming.
  - Patched torch.load(..., weights_only=False) for PyTorch 2.7 compatibility.

### Pipeline Documentation And Skill

- Time: `2026-03-17 06:36:56`
- Goal: Create a persistent markdown pipeline record, a reusable append helper, and a Codex skill that keeps future DynamicPoint workflow steps logged.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/append_pipeline_log.py`
  - `/root/.codex/skills/dynamicpoint-pipeline-recorder/SKILL.md`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/append_pipeline_log.py --doc /mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md --title "Step title" --goal "What changed" ...
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md`
  - `/root/.codex/skills/dynamicpoint-pipeline-recorder/SKILL.md`
- Note(s):
  - The skill is intended for future DynamicPoint or 4D preprocessing turns where reproducibility and command logging matter.
  - Future steps should append to the same markdown file instead of creating separate notes.

### Dilated LaMa Input

- Time: `2026-03-17 07:46:02`
- Goal: Generate a new LaMa input directory with SAM3 person masks dilated by 7 pixels to remove more edge residue around the person.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py`
- Command(s):
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py --images /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left --masks /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2 --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 --prefix left --dilate_pixels 7
```
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_lama_inpainting_input.py --images /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --masks /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --outdir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 --prefix right --dilate_pixels 7
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7`
- Note(s):
  - dilate_pixels=7 means a morphological dilation radius of 7 pixels using an elliptical kernel.

### LaMa Inpainting Output Dilate7

- Time: `2026-03-17 07:53:02`
- Goal: Run LaMa inpainting again using the dilated 7-pixel masks and prepare a separate output set for comparison.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/lama && PYTHONPATH=/mnt/d/develop/master_thesis /root/miniconda3/envs/sam3/bin/python bin/predict.py model.path=/mnt/d/develop/master_thesis/big-lama indir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 outdir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7 dataset.img_suffix=.jpg device=cuda:0
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7`
  - `/mnt/d/develop/4D/submodules/lama/big-lama`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/images`
- Note(s):
  - This run uses the same LaMa model and code patches as the previous run, but with 7-pixel dilated masks.

### LaMa Refine Dilate7

- Time: `2026-03-17 07:59:42`
- Goal: Run LaMa with refine=True on the 7-pixel dilated masks to improve boundary and structure quality.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/lama && PYTHONPATH=/mnt/d/develop/master_thesis /root/miniconda3/envs/sam3/bin/python bin/predict.py model.path=/mnt/d/develop/master_thesis/big-lama indir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 outdir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine dataset.img_suffix=.jpg device=cuda:0 refine=True refiner.gpu_ids=0,
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7`
  - `/mnt/d/develop/4D/submodules/lama/big-lama`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_refine/images`
- Note(s):
  - This run keeps the dilated masks and enables LaMa refinement on GPU 0.

### LaMa Refine Dilate7 Fast

- Time: `2026-03-17 08:02:37`
- Goal: Run a practical fast refinement variant of LaMa on the 7-pixel dilated masks by reducing refinement iterations, scales, and pixel budget.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/lama && PYTHONPATH=/mnt/d/develop/master_thesis /root/miniconda3/envs/sam3/bin/python bin/predict.py model.path=/mnt/d/develop/master_thesis/big-lama indir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 outdir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine_fast dataset.img_suffix=.jpg device=cuda:0 refine=True refiner.n_iters=2 refiner.max_scales=2 refiner.px_budget=600000
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7`
  - `/mnt/d/develop/4D/submodules/lama/big-lama`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine_fast`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_refine_fast/images`
- Note(s):
  - Abandoned the default refine configuration because it was too slow for 40 frames on this machine.
  - Single-GPU fallback for refiner.gpu_ids is handled inside predict.py.

### LaMa Refine Dilate7 Tiny

- Time: `2026-03-17 08:04:28`
- Goal: Run an aggressively downscaled refinement test of LaMa on the 7-pixel dilated masks to see whether refinement helps at all under the 8GB GPU limit.
- Script(s):
  - `/mnt/d/develop/4D/submodules/lama/bin/predict.py`
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/assemble_lama_inpainted_input.py`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/lama && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/mnt/d/develop/master_thesis /root/miniconda3/envs/sam3/bin/python bin/predict.py model.path=/mnt/d/develop/master_thesis/big-lama indir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7 outdir=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine_tiny dataset.img_suffix=.jpg device=cuda:0 refine=True refiner.n_iters=1 refiner.max_scales=1 refiner.px_budget=300000
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_input_dilate7`
  - `/mnt/d/develop/4D/submodules/lama/big-lama`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7_refine_tiny`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_refine_tiny/images`
- Note(s):
  - This run further reduces refinement cost after the previous refine attempt hit CUDA OOM on an 8GB GPU.

### COLMAP On LaMa Dilate7

- Time: `2026-03-17 08:19:24`
- Goal: Run COLMAP sparse reconstruction on the dilate7 LaMa-cleaned left/right image set.
- Script(s):
  - `/usr/bin/colmap`
- Command(s):
```bash
colmap feature_extractor --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/database.db --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --SiftExtraction.use_gpu 0
```
```bash
colmap exhaustive_matcher --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/database.db --SiftMatching.use_gpu 0
```
```bash
colmap mapper --database_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/database.db --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/images --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/sparse
```
```bash
colmap model_analyzer --path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/sparse/0
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/images`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/database.db`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/sparse/0`
- Note(s):
  - This reconstruction uses the inpainted dilate7 image set instead of the original left/right images.

### 3DGS Convert On LaMa Dilate7 Recon

- Time: `2026-03-17 08:25:03`
- Goal: Convert the COLMAP reconstruction from the LaMa dilate7-cleaned scene into the standard Gaussian Splatting training layout.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/convert.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/convert.py -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon --skip_matching --no_gpu
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/sparse/0`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/input`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/distorted`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/stereo`
- Note(s):
  - This is a standard Gaussian Splatting directory conversion, not the semantic/POI selective pipeline of CoRe-GS.

### Manual 3DGS Layout On LaMa Dilate7 Recon

- Time: `2026-03-17 08:26:31`
- Goal: Manually undistort the COLMAP reconstruction from the LaMa dilate7-cleaned scene and produce a usable Gaussian Splatting training layout when convert.py wrapper did not complete cleanly.
- Script(s):
  - `/usr/bin/colmap`
- Command(s):
```bash
colmap image_undistorter --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/input --input_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/distorted/sparse/0 --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon --output_type COLMAP
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/distorted/sparse/0`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/images`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/stereo`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/run-colmap-geometric.sh`
- Note(s):
  - convert.py wrapper complained on this directory layout, but direct COLMAP image_undistorter succeeded.
  - The usable sparse model for training is under sparse/0.

### 3DGS Environment Probe

- Time: `2026-03-17 08:34:25`
- Goal: Verified gaussian-splatting dependencies and selected sam3 conda environment for training.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python -c 'import torch, torchvision, simple_knn, diff_gaussian_rasterization'
```
- Input(s):
  - `/root/miniconda3/envs/sam3`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting`
- Note(s):
  - sam3 environment already contains torch 2.7.0+cu126, torchvision 0.22.0+cu126, simple_knn and diff_gaussian_rasterization; default system python lacks torch.

### 3DGS Input Fix

- Time: `2026-03-17 08:37:00`
- Goal: Fixed COLMAP camera height mismatch from 941 to 940, reran image undistortion, and produced a clean 3DGS-ready PINHOLE dataset.
- Script(s):
  - `/usr/bin/colmap`
- Command(s):
```bash
colmap model_converter --input_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/sparse/0 --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_camera_fixed_txt --output_type TXT && edit cameras.txt height 941->940 && colmap model_converter --input_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_camera_fixed_txt --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_camera_fixed_bin --output_type BIN && colmap image_undistorter --image_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7/images --input_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_camera_fixed_bin --output_path /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput --output_type COLMAP
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_recon/sparse/0`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput`
- Note(s):
  - LaMa inpainted PNGs are 1256x940 while the COLMAP sparse model stored height 941. After fixing the sparse camera height, image_undistorter succeeded and produced a PINHOLE camera with undistorted image height 939.

### 3DGS Training 7k

- Time: `2026-03-17 08:42:22`
- Goal: Trained a standard Gaussian Splatting model on the cleaned static-scene dataset for 7000 iterations.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2 --data_device cpu --iterations 7000 --test_iterations -1 --save_iterations 3000 5000 --disable_viewer
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2`
- Note(s):
  - Training completed successfully in about 4m51s on the RTX 4070 Laptop GPU using the sam3 environment. Saved Gaussian checkpoints at iterations 3000, 5000, and 7000.

### 3DGS Render 7k

- Time: `2026-03-17 08:43:50`
- Goal: Rendered all train views from the 7000-iteration Gaussian model for qualitative inspection.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2 --iteration 7000 --skip_test --quiet
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/train/ours_7000`
- Note(s):
  - Rendered 40 train views; outputs are under train/ours_7000/renders with matching ground truth under train/ours_7000/gt.

### 3DGS Evaluation Snapshot

- Time: `2026-03-17 08:44:25`
- Goal: Computed a quick qualitative/quantitative snapshot for the 7000-iteration Gaussian model.
- Script(s):
  - `/usr/bin/python3`
- Command(s):
```bash
python3 quick_eval_contact_sheet.py on /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/train/ours_7000
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/train/ours_7000`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/train/ours_7000/comparison_contact_sheet.png`
- Note(s):
  - Average train-view PSNR is 32.78 dB across 40 rendered views. Final Gaussian model contains 198355 vertices in point_cloud/iteration_7000/point_cloud.ply.

### FastGS Clone

- Time: `2026-03-17 08:46:30`
- Goal: Cloned the FastGS repository for accelerated Gaussian Splatting comparison on the same dataset.
- Script(s):
  - `/usr/bin/git`
- Command(s):
```bash
git clone https://github.com/fastgs/FastGS.git /mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS
```
- Input(s):
  - `https://github.com/fastgs/FastGS.git`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS`
- Note(s):
  - Repository cloned from the main branch. README indicates the static FastGS pipeline uses train.py with its own diff-gaussian-rasterization_fastgs CUDA extension.

### FastGS Rasterizer Install

- Time: `2026-03-17 08:49:11`
- Goal: Installed FastGS CUDA rasterizer extension into the sam3 environment.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/submodules/diff-gaussian-rasterization_fastgs/setup.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && cd /mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/submodules/diff-gaussian-rasterization_fastgs && python -m pip install --no-build-isolation .
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/submodules/diff-gaussian-rasterization_fastgs`
- Output(s):
  - `/root/miniconda3/envs/sam3/lib/python3.12/site-packages/diff_gaussian_rasterization_fastgs`
- Note(s):
  - Build isolation had to be disabled so the setup script could see torch from the sam3 conda environment.

### FastGS Runtime Dependency

- Time: `2026-03-17 08:49:44`
- Goal: Installed the missing websockets package required by FastGS network_gui_ws import path.
- Script(s):
  - `/usr/bin/pip`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python -m pip install websockets
```
- Input(s):
  - `/root/miniconda3/envs/sam3`
- Output(s):
  - `/root/miniconda3/envs/sam3/lib/python3.12/site-packages/websockets`
- Note(s):
  - FastGS imports gaussian_renderer.network_gui_ws at module import time even when websockets mode is not used, so the dependency must be present.

### FastGS Training 7k

- Time: `2026-03-17 08:51:48`
- Goal: Trained FastGS on the same static-scene dataset for 7000 iterations to compare speed and quality against vanilla 3DGS.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/train.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/train.py -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_fastgs_7k --data_device cpu --iterations 7000 --test_iterations -1 --save_iterations 3000 5000 --checkpoint_iterations 7000 --densification_interval 500 --optimizer_type default --grad_abs_thresh 0.0012
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_fastgs_7k`
- Note(s):
  - FastGS finished 7000 iterations in 73.277s and produced 68498 Gaussians at the final checkpoint.

### FastGS Render 7k

- Time: `2026-03-17 08:53:32`
- Goal: Rendered FastGS train views and computed a quick PSNR snapshot for comparison against vanilla 3DGS.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/render.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python /mnt/d/develop/master_thesis/DynamicPoint/submodules/FastGS/render.py -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_fastgs_7k --iteration 7000 --skip_test --quiet
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_fastgs_7k`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_fastgs_7k/train/ours_7000`
- Note(s):
  - Rendered 40 train views. A follow-up quick evaluation computes train-view PSNR and writes comparison_contact_sheet.png in the same folder.

### DreamScene4D Dataset Prep

- Time: `2026-03-17 09:00:24`
- Goal: Prepared left and right single-view sequences plus person masks in DreamScene4D JPEGImages/Annotations format.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_dreamscene4d_dataset.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_dreamscene4d_dataset.py --image_dir <left_or_right> --mask_dir <matching_mask_dir> --output_root /mnt/d/develop/4D/submodules/dreamscene4d/data --video_name <forest_walk_left_or_right>
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left`
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right`
- Output(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_left`
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right`
- Note(s):
  - Prepared two single-view DreamScene4D datasets. Each one contains 20 RGB frames under JPEGImages and one object track under Annotations/001.

### DreamScene4D Submodules

- Time: `2026-03-17 09:01:29`
- Goal: Initialized DreamScene4D git submodules required for GMFlow and modified diff-gaussian-rasterization.
- Script(s):
  - `/usr/bin/git`
- Command(s):
```bash
cd /mnt/d/develop/4D/submodules/dreamscene4d && git submodule update --init --recursive
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/.gitmodules`
- Output(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/gmflow`
  - `/mnt/d/develop/4D/submodules/dreamscene4d/diff-gaussian-rasterization`
- Note(s):
  - DreamScene4D requires its submodules to be present before dependency installation and training.

### DreamScene4D Environment Check

- Time: `2026-03-17 09:22:57`
- Goal: Audit DreamScene4D runtime readiness in the sam3 conda environment before training on the prepared monocular datasets.
- Script(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/README.md`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python import checks; cd /mnt/d/develop/4D/submodules/dreamscene4d && git submodule status; nvidia-smi
```
- Input(s):
  - `/root/miniconda3/envs/sam3;/mnt/d/develop/4D/submodules/dreamscene4d;/mnt/d/develop/4D/submodules/dreamscene4d/requirements.txt`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/docs/forest_walk_0806_0810_pipeline.md`
- Note(s):
  - Ready: torch/cuda, simple_knn, diff_gaussian_rasterization, most Python deps. Blockers: local diffusers import fails against huggingface_hub 1.4.1 (hf_cache_home missing), main/main_4d/main_4d_compose import therefore fail, gmflow pretrained checkpoint missing, nvdiffrast not installed, mcubes binary ABI mismatch with numpy 2.4.2. Repo README recommends a separate Python 3.8 env with torch 2.2/cu118, while current env is Python 3.12 + torch 2.7/cu126 on RTX 4070 Laptop 8GB.

### DreamScene4D Quick Training

- Time: `2026-03-17 10:22:38`
- Goal: Run a reduced-iteration DreamScene4D pipeline on the prepared forest_walk_right monocular sequence to obtain a quick end-to-end dynamic Gaussian result on 8GB VRAM.
- Script(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/main.py;/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py;/mnt/d/develop/4D/submodules/dreamscene4d/main_4d_compose.py`
- Command(s):
```bash
python main.py --config configs/image.yaml input=./data/JPEGImages/forest_walk_right/00000.png input_mask=./data/Annotations/forest_walk_right/001/00000.png outdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians visdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/vis save_path=forest_walk_right_quick_1 iters=80; python main_4d.py --config configs/4d.yaml input=./data/JPEGImages/forest_walk_right input_mask=./data/Annotations/forest_walk_right/001 outdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians visdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/vis save_path=forest_walk_right_quick_1 iters=40 batch_size=1 n_views=1; python main_4d_compose.py --config configs/4d.yaml input=./data/JPEGImages/forest_walk_right input_mask='[./data/Annotations/forest_walk_right/001/00000.png]' outdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians visdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/vis save_path=forest_walk_right_quick iters=40 batch_size=1 n_views=1
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right;/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right/001;/mnt/d/develop/4D/submodules/dreamscene4d/gmflow/pretrained/gmflow_kitti-285701a8.pth`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/vis`
- Note(s):
  - Official run_no_inpaint.py --lite was started first but was too slow for this GPU because stage 1 still used 1000 iterations. Switched to manual three-stage quick settings. Stage 2 initially OOMed with default batch_size=10 and n_views=4 on 8GB VRAM; reran successfully with batch_size=1 and n_views=1. Compose completed and saved forest_walk_right_quick_trajs.mp4.

### DreamScene4D Environment Setup

- Time: `2026-03-17 10:22:38`
- Goal: Create a dedicated DreamScene4D conda environment, install repo dependencies, compile local CUDA extensions, and prepare GMFlow weights for monocular dynamic Gaussian training.
- Script(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/README.md`
- Command(s):
```bash
conda create -n dreamscene4d python=3.8.18; conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia; conda install -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8.89 cuda-compiler=11.8.0 cuda-libraries-dev=11.8.0; pip install ./diffusers -r requirements.txt gdown; pip install ./simple-knn ./diff-gaussian-rasterization git+https://github.com/NVlabs/nvdiffrast/; gdown GMFlow zip and extract pretrained/gmflow_kitti-285701a8.pth
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d;/mnt/d/develop/4D/submodules/dreamscene4d/gmflow;/root/miniconda3/envs/dreamscene4d`
- Output(s):
  - `/root/miniconda3/envs/dreamscene4d;/mnt/d/develop/4D/submodules/dreamscene4d/gmflow/pretrained/gmflow_kitti-285701a8.pth`
- Note(s):
  - Dedicated env created to avoid conflicts with sam3. Added minimal local patches so simple-knn and diff-gaussian-rasterization compile against conda CUDA 11.8 headers. gdown delivered a zip bundle from Google Drive; extracted the real GMFlow checkpoint and kept the original archive as gmflow_kitti-285701a8.zipbackup.
