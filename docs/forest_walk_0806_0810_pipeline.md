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

### Prepare DreamScene4D Raw-Mask Right Dataset

- Time: `2026-03-17 22:37:02`
- Goal: Create a separate DreamScene4D input sequence that explicitly uses the original non-dilated SAM3 right-view masks for a clean rerun.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_dreamscene4d_dataset.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_dreamscene4d_dataset.py --image_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --mask_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --output_root /mnt/d/develop/4D/submodules/dreamscene4d/data --video_name forest_walk_right_rawmask
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person`
- Output(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right_rawmask;/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right_rawmask/001`
- Note(s):
  - The original prepare_dreamscene4d_dataset.py script only binarizes masks and does not dilate them. This rerun uses the raw SAM3 masks under a new video name to avoid ambiguity with earlier forest_walk_right outputs.

### Verify DreamScene4D Used Raw Masks

- Time: `2026-03-17 22:39:50`
- Goal: Confirm whether the earlier DreamScene4D run used non-dilated right-view SAM3 masks before spending more time on a duplicate rerun.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_dreamscene4d_dataset.py`
- Command(s):
```bash
python3 verification scripts comparing /mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right/001 against /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person; partial rerun forest_walk_right_rawmask_quick was stopped after confirming identical masks
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right/001;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right`
- Output(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right_rawmask;/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right_rawmask/001`
- Note(s):
  - prepare_dreamscene4d_dataset.py only binarizes masks. Pixel-count checks for frames 0, 5, 10, and 19 matched exactly between DreamScene4D input masks and right_masks_person, and all 20 annotation files matched byte-for-byte. A duplicate raw-mask retrain was started but stopped once this was confirmed, because it would reproduce the earlier forest_walk_right quick run.

### Auto Align DreamScene4D To Static Scene

- Time: `2026-03-23 00:10:28`
- Goal: Estimate a reproducible automatic Sim(3) transform that places the DreamScene4D right-view dynamic object into the static COLMAP world using multi-frame 2D bbox constraints.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamscene4d && python /mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py --colmap_model /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0 --image_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --mask_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --dynamic_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d.pkl --dynamic_motion_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d_global_motion.pkl --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_4d_model.ply --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right/aligned_dynamic_frame0.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right/overlay_contact_sheet.png`
- Note(s):
  - This is a coarse automatic alignment, not a final registration. It uses five frames (0,5,10,15,19), COLMAP right-view cameras, DreamScene4D per-frame global motion, and bbox projection residuals instead of ICP. The resulting transform is suitable as a reproducible initialization for later manual refinement or local ICP.

### Auto Align DreamScene4D To Static Scene (All 20 Frames)

- Time: `2026-03-23 00:12:20`
- Goal: Test whether using all 20 right-view frames for bbox-based Sim(3) optimization improves dynamic-to-static alignment stability over the 5-frame initialization run.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamscene4d && python /mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py --colmap_model /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0 --image_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --mask_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --dynamic_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d.pkl --dynamic_motion_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d_global_motion.pkl --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_4d_model.ply --frame_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/aligned_dynamic_frame0.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/overlay_contact_sheet.png`
- Note(s):
  - The all-frame run converged, but is intended for comparison against the cheaper 5-frame coarse alignment. Use the metric summary to decide which initialization is more useful before adding stronger constraints such as left-view consistency, foot-point anchoring, or local ICP.

### Export Auto-Alignment Overlay Samples

- Time: `2026-03-23 00:14:36`
- Goal: Create a compact preview image from representative auto-alignment overlay frames for quick visual inspection.
- Script(s):
  - `/usr/bin/python3`
- Command(s):
```bash
python3 build preview from /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/overlays frames 0,5,10,15,19 into overlay_samples_all20.png
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/overlays`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/overlay_samples_all20.png`
- Note(s):
  - This preview reuses the generated overlay frames. Green box is the observed SAM3 mask bbox and red box is the projected dynamic model bbox after automatic alignment.

## TODO

- Integrate the camera poses and depth estimation method provided by VIPE into both the static and dynamic pipelines when time allows, and compare the reconstruction/alignment quality against the current DreamScene4D default-camera setup and the existing COLMAP-based static pipeline.

### Render Registered Novel-View Preview

- Time: `2026-03-23 01:06:36`
- Goal: Generate a quick novel-view preview of the current registration state by rendering the static sparse background point cloud together with the aligned dynamic DreamScene4D snapshot.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py --static_ply /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0/points3D.ply --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/aligned_dynamic_snapshot_from_ply.ply --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0/points3D.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/aligned_dynamic_snapshot_from_ply.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview/novel_view_contact_sheet.png;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview/novel_view_orbit.mp4`
- Note(s):
  - This is a fast point-cloud preview, not a full Gaussian render. It visualizes the static sparse background and the aligned dynamic snapshot together from multiple azimuths to judge spatial registration quality.

### Render Registered Novel-View Preview With Static 3DGS Background

- Time: `2026-03-23 01:10:31`
- Goal: Generate a cleaner novel-view preview using the static 3DGS Gaussian point cloud instead of the sparse COLMAP point cloud, together with the aligned dynamic snapshot.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py --static_ply /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/point_cloud/iteration_7000/point_cloud.ply --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/aligned_dynamic_snapshot_from_ply.ply --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview_gs
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/point_cloud/iteration_7000/point_cloud.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/aligned_dynamic_snapshot_from_ply.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview_gs/novel_view_contact_sheet.png;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/novel_view_preview_gs/novel_view_orbit.mp4`
- Note(s):
  - This preview decodes static Gaussian colors from f_dc and uses opacity/scale as a lightweight visualization prior. It is still a point-cloud preview rather than a full differentiable Gaussian render, but it produces a cleaner background than sparse COLMAP points for judging registration.

### Merge Static And Dynamic Gaussian Splats For SuperSplat

- Time: `2026-03-23 01:13:47`
- Goal: Create a single Gaussian PLY that combines the static 3DGS background with the globally aligned DreamScene4D dynamic object so it can be inspected in a real splat viewer such as SuperSplat.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/merge_registered_gaussians.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/merge_registered_gaussians.py --static_ply /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/point_cloud/iteration_7000/point_cloud.ply --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_4d_model.ply --transform_json /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/transform.json --output_ply /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/merged_static_dynamic_supersplat.ply
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/point_cloud/iteration_7000/point_cloud.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_4d_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/transform.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_all20/merged_static_dynamic_supersplat.ply`
- Note(s):
  - Dynamic DreamScene4D Gaussians were transformed into the static 3DGS world using the all20 Sim(3) transform. Missing f_rest_* channels in the dynamic PLY were filled with zeros so the merged file matches the static 3DGS schema expected by splat viewers.

### Clone VIPE

- Time: `2026-03-23 01:22:56`
- Goal: Bring the official VIPE repository into the local workspace to evaluate replacing or augmenting the current camera-pose/depth parts of the pipeline.
- Script(s):
  - `git`
- Command(s):
```bash
git clone https://github.com/nv-tlabs/vipe.git /mnt/d/develop/master_thesis/external/vipe
```
- Input(s):
  - `https://github.com/nv-tlabs/vipe`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/vipe`
- Note(s):
  - Cloned the public official VIPE repository from NVIDIA as a candidate replacement for the current default-camera and depth-estimation setup.

### Create VIPE conda env

- Time: `2026-03-23 01:30:47`
- Goal: Create a dedicated conda environment for NVIDIA ViPE before installing repo pip dependencies and running pose/depth inference on the forest_walk right-view sequence.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/envs/base.yml`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda env create -f /mnt/d/develop/master_thesis/external/vipe/envs/base.yml
```
- Input(s):
  - `/mnt/d/develop/master_thesis/external/vipe/envs/base.yml`
- Output(s):
  - `/root/miniconda3/envs/vipe`
- Note(s):
  - base.yml only creates the Python 3.10 conda env. ViPE Python packages and CUDA wheels still need pip installation from envs/requirements.txt and editable install of the repo.

### Install VIPE pip dependencies

- Time: `2026-03-23 01:54:50`
- Goal: Install the official ViPE Python dependencies and editable package into the dedicated vipe conda environment so the right-view sequence can be processed for pose/depth estimation.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/envs/requirements.txt`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cd /mnt/d/develop/master_thesis/external/vipe && pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 && pip install --no-build-isolation -e .
```
- Input(s):
  - `/mnt/d/develop/master_thesis/external/vipe/envs/requirements.txt;/mnt/d/develop/master_thesis/external/vipe/pyproject.toml`
- Output(s):
  - `/root/miniconda3/envs/vipe;/mnt/d/develop/master_thesis/external/vipe`
- Note(s):
  - Installing ViPE pulls a full CUDA 12.8 PyTorch stack and large auxiliary wheels. This step is heavy but isolated to the dedicated vipe environment and does not modify existing DynamicPoint or DreamScene4D environments.

### Fix VIPE CUDA headers and JIT-build extension

- Time: `2026-03-23 02:08:38`
- Goal: Make the ViPE extension compile inside the dedicated vipe environment by exposing conda CUDA headers in $CONDA_PREFIX/include and using the source tree with JIT extension build.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/ext/__init__.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cp -asn $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/ && export PYTHONPATH=/mnt/d/develop/master_thesis/external/vipe:$PYTHONPATH && export VIPE_EXT_JIT=1 && export MAX_JOBS=1 && python -c "import vipe; print(vipe.__file__)"
```
- Input(s):
  - `/root/miniconda3/envs/vipe/targets/x86_64-linux/include;/mnt/d/develop/master_thesis/external/vipe`
- Output(s):
  - `/root/miniconda3/envs/vipe/include;/root/.cache/torch_extensions/py310_cu128/vipe_ext_jit`
- Note(s):
  - Editable installation failed because torch extension compilation did not see CUDA headers. Symlinking conda target headers into $CONDA_PREFIX/include and forcing single-job JIT compilation allowed the source checkout to import successfully without a formal pip install of the vipe package.

### Patch VIPE JIT extension alias

- Time: `2026-03-23 02:23:07`
- Goal: Allow ViPE post-processing code paths that import vipe_ext directly to work when the environment uses the JIT-built vipe_ext_jit module instead of a compiled package installation.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/ext/__init__.py`
- Command(s):
```bash
apply minimal local patch: register sys.modules['vipe_ext'] = _C after loading vipe_ext_jit
```
- Input(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/ext/__init__.py`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/ext/__init__.py`
- Note(s):
  - The default code exposes the JIT-built extension as vipe_ext_jit only. PriorDA depth completion later imports vipe_ext directly, which caused ModuleNotFoundError during save_artifacts. The patch keeps behavior unchanged for packaged installs and only adds the missing alias for the JIT fallback path.

### Run VIPE pose-only baseline on right view

- Time: `2026-03-23 02:36:12`
- Goal: Produce a faster VIPE baseline for the right-view sequence by keeping pose/intrinsics/SLAM map generation but skipping the expensive final depth-alignment stage.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/run.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cd /mnt/d/develop/master_thesis/external/vipe && export PYTHONPATH=/mnt/d/develop/master_thesis/external/vipe:$PYTHONPATH && export VIPE_EXT_JIT=1 && export MAX_JOBS=1 && python run.py pipeline=default streams=frame_dir_stream streams.base_path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right pipeline.post.depth_align_model=null pipeline.output.path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly pipeline.output.save_artifacts=true pipeline.output.save_slam_map=true pipeline.output.save_viz=false
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/external/vipe/configs/pipeline/default.yaml`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly`
- Note(s):
  - The full default VIPE pipeline on 20 right-view frames spent several minutes in final depth alignment without incremental artifact flush. This pose-only baseline retains pose, intrinsics, RGB, and optional SLAM map outputs while skipping the slow post depth alignment so the results can be compared sooner.

### Patch VIPE RGB writer for odd image sizes

- Time: `2026-03-23 02:38:55`
- Goal: Allow VIPE artifact persistence on 1256x941 right-view frames by padding frames to even dimensions before libx264 encoding.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/utils/visualization.py`
- Command(s):
```bash
apply minimal local patch: pad odd-height/odd-width frames by one pixel in VideoWriter.write before appending to imageio/libx264
```
- Input(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/utils/visualization.py`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/vipe/vipe/utils/visualization.py`
- Note(s):
  - Without this patch VIPE saved pose/intrinsics first, then failed on rgb/right.mp4 because libx264 yuv420p rejects 1256x941. The patch preserves content and only pads the bottom/right border when needed.

### Persist VIPE pose-only artifacts

- Time: `2026-03-23 02:40:54`
- Goal: Finish a practical VIPE baseline run that persists reusable intermediate outputs for the right-view sequence, including camera pose, intrinsics, RGB video, and SLAM map.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/run.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cd /mnt/d/develop/master_thesis/external/vipe && export PYTHONPATH=/mnt/d/develop/master_thesis/external/vipe:$PYTHONPATH && export VIPE_EXT_JIT=1 && export MAX_JOBS=1 && python run.py pipeline=default streams=frame_dir_stream streams.base_path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right pipeline.post.depth_align_model=null pipeline.output.path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2 pipeline.output.save_artifacts=true pipeline.output.save_slam_map=true pipeline.output.save_viz=false
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/external/vipe/configs/pipeline/default.yaml`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2`
- Note(s):
  - This successful baseline disables the expensive final depth alignment to keep runtime manageable. Persisted artifacts include pose/right.npz, intrinsics/right.npz, intrinsics/right_camera.txt, rgb/right.mp4, and vipe/right_slam_map.pt. It does not produce depth/right.zip because pipeline.post.depth_align_model=null.

### Start VIPE no_vda depth run in background

- Time: `2026-03-23 02:41:26`
- Goal: Launch a depth-persisting VIPE run with the lighter no_vda configuration so camera pose, intrinsics, RGB, SLAM map, and depth can all be persisted without the very slow default SVDA depth alignment.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/run.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cd /mnt/d/develop/master_thesis/external/vipe && export PYTHONPATH=/mnt/d/develop/master_thesis/external/vipe:$PYTHONPATH && export VIPE_EXT_JIT=1 && export MAX_JOBS=1 && nohup python run.py pipeline=no_vda streams=frame_dir_stream streams.base_path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right pipeline.output.path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_novda pipeline.output.save_artifacts=true pipeline.output.save_slam_map=true pipeline.output.save_viz=false > /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_novda_run.log 2>&1 &
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/external/vipe/configs/pipeline/no_vda.yaml`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_novda;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_novda_run.log`
- Note(s):
  - This run was started in the background to continue depth persistence while preserving the already successful pose-only persisted baseline. Monitor progress via the nohup log file.

### Convert VIPE Poses For DreamScene4D

- Time: `2026-03-23 02:51:12`
- Goal: Persist DreamScene4D-compatible camera pose JSON and the missing cam_scales artifact from the VIPE pose-only run so stage 2 can consume external camera motion.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/vipe_pose_to_dreamscene_cam_pose.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/vipe_pose_to_dreamscene_cam_pose.py --vipe_pose_npz /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/pose/right.npz --output_json /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json --output_cam_scales /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_cam_scales.npy --default_scale 1.0
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/pose/right.npz`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_cam_scales.npy`
- Note(s):
  - VIPE pose artifacts are OpenCV cam2world matrices, converted here to DreamScene4D JSON entries with pos plus xyzw quaternion orientation. DreamScene4D does not generate cam_scales internally on this code path, so this experiment seeds a conservative all-ones cam_scales array to make external pose loading runnable.

### Reuse DreamScene4D Stage1 For VIPE-Pose Experiment

- Time: `2026-03-23 03:15:09`
- Goal: Isolate the effect of external VIPE camera poses by reusing the previously trained stage-1 DreamScene4D initialization instead of rerunning an equivalent stage that does not consume cam_pose.
- Script(s):
  - `/bin/cp`
- Command(s):
```bash
mkdir -p /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis && cp /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_model.ply /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_model.ply && cp /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1.pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1.pkl && cp /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_global_motion.pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_global_motion.pkl
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/forest_walk_right_quick_1_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_quick/gaussians/gaussians/forest_walk_right_quick_1_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_global_motion.pkl`
- Note(s):
  - A direct stage-1 rerun was started but abandoned because it does not use cam_pose and its runtime grew far beyond the earlier quick baseline. Reusing the old stage-1 assets keeps the comparison focused on the stage-2/compose camera-motion change.

### DreamScene4D Stage2 With VIPE Camera Poses

- Time: `2026-03-23 03:26:39`
- Goal: Re-train the DreamScene4D dynamic stage using external VIPE camera motion instead of the default identity per-frame camera setup.
- Script(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/main_4d.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamscene4d && cd /mnt/d/develop/4D/submodules/dreamscene4d && python main_4d.py --config configs/4d.yaml input=./data/JPEGImages/forest_walk_right input_mask=./data/Annotations/forest_walk_right/001 cam_pose=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json outdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians visdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis save_path=forest_walk_right_vipepose_quick_1 iters=40 batch_size=1 n_views=1
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right;/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right/001;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_cam_scales.npy`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_4d_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Note(s):
  - Stage 1 was intentionally reused from the earlier quick experiment because it does not consume cam_pose. External VIPE camera poses were loaded successfully together with a conservative all-ones cam_scales array, so this run isolates the effect of camera motion in stage 2.

### DreamScene4D Compose With VIPE Camera Poses

- Time: `2026-03-23 03:27:58`
- Goal: Render the composed DreamScene4D visualization after stage-2 training with external VIPE camera poses.
- Script(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/main_4d_compose.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamscene4d && cd /mnt/d/develop/4D/submodules/dreamscene4d && python main_4d_compose.py --config configs/4d.yaml input=./data/JPEGImages/forest_walk_right input_mask='[./data/Annotations/forest_walk_right/001/00000.png]' cam_pose=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json outdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians visdir=/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis save_path=forest_walk_right_vipepose_quick iters=40 batch_size=1 n_views=1
```
- Input(s):
  - `/mnt/d/develop/4D/submodules/dreamscene4d/data/JPEGImages/forest_walk_right;/mnt/d/develop/4D/submodules/dreamscene4d/data/Annotations/forest_walk_right/001/00000.png;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/dreamscene/right_cam_pose.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis/forest_walk_right_vipepose_quick_composed_no_orbit.gif;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis/forest_walk_right_vipepose_quick_composed_hor_orbit.gif;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis/forest_walk_right_vipepose_quick_composed_elev_orbit.gif;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/vis/forest_walk_right_vipepose_quick_trajs.mp4`
- Note(s):
  - Compose accepted the same VIPE-derived cam_pose JSON. DreamScene4D still estimates depth internally via Depth Anything; only the camera motion source changed in this experiment.

### Convert VIPE Pose+SLAM To COLMAP For Static Right Sequence

- Time: `2026-03-23 03:35:48`
- Goal: Export the VIPE right-view pose-only reconstruction, augmented with SLAM-map points, into a COLMAP text model for a single-view static reconstruction experiment.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/scripts/vipe_to_colmap.py`
- Command(s):
```bash
python /mnt/d/develop/master_thesis/external/vipe/scripts/vipe_to_colmap.py /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2 --sequence right --use_slam_map --output /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/pose/right.npz;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/intrinsics/right.npz;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/vipe/right_slam_map.pt;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/rgb/right.mp4`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/cameras.txt;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/images.txt;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/points3D.txt;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/images`
- Note(s):
  - The upstream converter hard-requires a depth artifact path even when --use_slam_map is selected, so an empty placeholder depth/right.zip was materialized purely to satisfy that existence check. Geometry for this export comes from the saved VIPE SLAM map, not from unprojected depth frames.

### Prepare VIPE-Based Static GS Input

- Time: `2026-03-23 03:36:17`
- Goal: Combine the VIPE-exported right-view COLMAP text model with LaMa-cleaned right-view frames to form a single-view static Gaussian Splatting scene directory.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_vipe_static_right_gs_input.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_vipe_static_right_gs_input.py --vipe_colmap_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right --lama_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7 --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput/images;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput/sparse/0`
- Note(s):
  - This static comparison intentionally uses only the right monocular view. The 20 LaMa-cleaned right images are converted to frame_XXXXXX.jpg so they match the VIPE-exported COLMAP image names.

### Fix VIPE Static GS Image Names

- Time: `2026-03-23 03:37:07`
- Goal: Normalize image names in the VIPE-exported images.txt so gaussian-splatting resolves the single-view right images correctly.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_vipe_static_right_gs_input.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/prepare_vipe_static_right_gs_input.py --vipe_colmap_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right --lama_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7 --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/images.txt`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput/sparse/0/images.txt`
- Note(s):
  - The upstream VIPE COLMAP exporter writes image names as images/frame_XXXXXX.jpg. Gaussian-splatting appends its own images directory, so the names must be normalized to plain frame_XXXXXX.jpg to avoid images/images/... lookup failures.

### Train Static 3DGS With VIPE Right-View Poses

- Time: `2026-03-23 03:41:55`
- Goal: Train a single-view static Gaussian Splatting model using VIPE camera poses and SLAM-map initialization on the LaMa-cleaned right-view sequence.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k --data_device cpu --iterations 7000 --test_iterations -1 --save_iterations 3000 5000 --disable_viewer
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_static_gsinput`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k/point_cloud/iteration_7000/point_cloud.ply`
- Note(s):
  - The same gaussians-splatting command line family as the earlier static baseline was reused, but with a monocular right-view VIPE scene instead of the left-right COLMAP scene. Training succeeded only in the sam3 environment because its diff_gaussian_rasterization build matches the current renderer API.

### Render Static VIPE-Based 3DGS

- Time: `2026-03-23 03:42:37`
- Goal: Render the training views of the VIPE-based static right-view Gaussian model for direct visual inspection.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k --iteration 7000 --skip_test --quiet
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k/train/ours_7000`
- Note(s):
  - This render pass uses the same sam3 environment as training because the rasterizer build must match the training binary interface.

### Run VIPE Pose-Only On Left View

- Time: `2026-03-23 03:52:48`
- Goal: Persist VIPE camera poses, intrinsics, masks, RGB video, and SLAM map for the left monocular sequence so left/right static fusion can be attempted.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/vipe/run.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vipe && cd /mnt/d/develop/master_thesis/external/vipe && python run.py pipeline=default streams=frame_dir_stream streams.base_path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left pipeline.post.depth_align_model=null pipeline.output.path=/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2 pipeline.output.save_artifacts=true pipeline.output.save_slam_map=true pipeline.output.save_viz=false
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2/pose/left.npz;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2/intrinsics/left.npz;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2/vipe/left_slam_map.pt`
- Note(s):
  - This matches the previously stabilized right-view pose-only configuration: VIPE JIT extension path, depth_align_model=null, save_artifacts=true, save_slam_map=true, and no visualization video.

### Merge Left/Right VIPE Static Scene

- Time: `2026-03-23 03:55:41`
- Goal: Align the left VIPE trajectory into the right VIPE world frame and build a single two-view COLMAP-style static scene from both VIPE reconstructions plus LaMa-cleaned images.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/build_merged_vipe_lr_static_scene.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/DynamicPoint/scripts/build_merged_vipe_lr_static_scene.py --right_pose_npz /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/pose/right.npz --left_pose_npz /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2/pose/left.npz --right_intrinsics_npz /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2/intrinsics/right.npz --left_intrinsics_npz /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2/intrinsics/left.npz --right_points3d_txt /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right/points3D.txt --left_points3d_txt /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2_colmap/left/points3D.txt --lama_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7 --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_right_poseonly_v2_colmap/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_left_poseonly_v2_colmap/left;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/lama_output_dilate7`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput/images;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput/vipe_lr_alignment.json`
- Note(s):
  - The left trajectory is aligned into the right world frame with a Sim(3) fit over synchronized camera centers. The resulting center RMSE is about 0.00498 and the mean left-vs-right orientation offset is about 58.9 degrees, which is consistent with the expected fixed crop-angle difference between the two virtual views.

### Train Static 3DGS With Merged Left+Right VIPE Scene

- Time: `2026-03-23 04:00:36`
- Goal: Train a two-view static Gaussian Splatting model using the merged left/right VIPE camera poses and merged SLAM-map initialization.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/train.py -s /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lr_vipe_gs_7k --data_device cpu --iterations 7000 --test_iterations -1 --save_iterations 3000 5000 --disable_viewer
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/vipe_lr_static_gsinput`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lr_vipe_gs_7k/point_cloud/iteration_7000/point_cloud.ply`
- Note(s):
  - This is the merged left+right VIPE static baseline. It uses two camera entries in cameras.txt, 40 images total, and the left SLAM-map geometry transformed into the right world frame before training.

### Render Merged Left+Right VIPE Static 3DGS

- Time: `2026-03-23 04:01:45`
- Goal: Render the training views of the merged left/right VIPE static Gaussian model for visual comparison against the earlier baselines.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && python /mnt/d/develop/master_thesis/DynamicPoint/submodules/gaussian-splatting/render.py -m /mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lr_vipe_gs_7k --iteration 7000 --skip_test --quiet
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lr_vipe_gs_7k`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lr_vipe_gs_7k/train/ours_7000`
- Note(s):
  - This render covers all 40 training views, including both left and right VIPE-aligned cameras.

### Auto Align VIPE-Pose DreamScene4D To COLMAP Static Scene

- Time: `2026-03-23 04:08:58`
- Goal: Test whether the DreamScene4D run driven by external VIPE camera poses improves coarse automatic alignment to the COLMAP-based static scene.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamscene4d && python /mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py --colmap_model /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0 --image_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right --mask_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person --dynamic_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl --dynamic_motion_pkl /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl --dynamic_ply /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_4d_model.ply --frame_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --output_dir /mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20
```
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/overlay_contact_sheet.png;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/overlay_samples_all20.png`
- Note(s):
  - This reruns the existing bbox-based Sim(3) alignment with the VIPE-pose DreamScene4D dynamic model while keeping the static COLMAP scene and right-view masks fixed. The resulting optimization cost decreased from about 0.5482 to about 0.5188 relative to the earlier default-camera DreamScene4D alignment, indicating a modest improvement but not a qualitative fix by itself.

### Summarize VIPE-pose dynamic alignment check

- Time: `2026-03-23 04:11:03`
- Goal: Assess whether VIPE camera poses improve coarse alignment between DreamScene4D dynamic output and the COLMAP static scene.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/overlay_samples_all20.png`
- Note(s):
  - Verified persisted VIPE-pose DreamScene4D alignment artifacts and compared objective cost against the prior default-camera DreamScene4D alignment run.

### Extend auto alignment with left and right view constraints

- Time: `2026-03-23 04:16:37`
- Goal: Improve DreamScene4D-to-static alignment by jointly optimizing against left and right view mask boxes instead of only the right view.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_lr_all20/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_lr_all20/overlay_contact_sheet.png`
- Note(s):
  - Updated auto alignment script to parse multiple views from COLMAP, accept per-view image and mask directories, and optimize a single Sim(3) against both left and right box observations.

### Run left-plus-right constrained dynamic alignment

- Time: `2026-03-23 04:18:06`
- Goal: Test whether jointly constraining the DreamScene4D dynamic model with left and right masks improves coarse alignment robustness over right-only optimization.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/left_masks_person_v2;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_4d_model.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_lr_all20/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_lr_all20/overlay_contact_sheet.png`
- Note(s):
  - The joint left+right optimization converged and produced 40 overlay images. Raw cost increased because the residual vector doubled; compare normalized residuals instead of raw cost when judging whether the added left-view constraints help.

### Add manual dynamic placement helper

- Time: `2026-03-23 04:22:16`
- Goal: Enable manual Sim(3) placement of the DreamScene4D dynamic model relative to the static background by editing scale, rotation, and translation directly.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/apply_manual_dynamic_transform.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_4d_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/transform.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/manual_place_dynamic/manual_transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/manual_place_dynamic/aligned_dynamic_snapshot_splat.ply`
- Note(s):
  - The helper script applies either an explicit Sim(3) or small deltas on top of an existing auto-alignment transform and exports both a frame-specific XYZ ply and a splat-compatible snapshot ply for manual inspection in CloudCompare or SuperSplat.

### Add foot-ground constraint to dynamic alignment

- Time: `2026-03-23 04:26:34`
- Goal: Reduce upside-down or floating placements by anchoring the lowest point of the dynamic model to the first-frame foot contact estimate in the static scene.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/colmap_pair_lama_dilate7_gsinput/sparse/0;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/views_scgs_4x3/right_masks_person;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20_foot/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20_foot/overlay_contact_sheet.png`
- Note(s):
  - Extended the optimizer with an estimated world-up vector and a first-frame foot anchor constraint based on the mask bottom center back-projected with the sparse depth hint.

### Compare foot-constrained and unconstrained keyframe alignment

- Time: `2026-03-23 04:28:22`
- Goal: Measure whether the newly added foot-ground term improves coarse right-view alignment on a matched five-keyframe subset.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/auto_align_dynamic_to_static.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_foot_keyframes/transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_keyframes_nfoots/transform.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_foot_keyframes/overlay_contact_sheet.png;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_keyframes_nfoots/overlay_contact_sheet.png`
- Note(s):
  - On the matched five-keyframe subset, the foot-constrained run ended with a higher normalized residual than the unconstrained run. The new term may still help uprightness qualitatively, but it does not improve the current bbox-based objective on its own.

### Export DreamScene4D as aligned PLY sequence

- Time: `2026-03-23 05:07:29`
- Goal: Create a per-frame PLY sequence suitable for SuperSplat 4DGS video export tests, using the current best automatic Sim(3) alignment as a shared transform across all frames.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/export_dreamscene4d_ply_sequence.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/transform.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_autoalign/sequence_meta.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_autoalign/forest_walk_right_vipepose_000000.ply`
- Note(s):
  - Each exported frame keeps the DreamScene4D Gaussian attributes and applies the per-frame global motion followed by the shared auto-alignment transform, so the whole sequence stays editable as one object if re-exported with a different base transform.

### Apply SuperSplat manual transform candidate

- Time: `2026-03-23 06:30:01`
- Goal: Test whether a manual transform estimated in SuperSplat improves the dynamic asset placement when reapplied uniformly outside the viewer.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/apply_manual_dynamic_transform.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/forest_walk_right_vipepose_quick_1_4d_model.ply;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/auto_align_dreamscene_right_vipepose_all20/transform.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/manual_place_dynamic_supersplat_try1/manual_transform.json;/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/manual_place_dynamic_supersplat_try1/aligned_dynamic_snapshot_splat.ply`
- Note(s):
  - Applied the viewer-side candidate delta rotation (0,0,-36 deg) and delta translation (16,-12,6) on top of the current best auto-alignment transform as a first manual placement experiment.

### Export first manual-placement PLY sequence candidate

- Time: `2026-03-23 06:30:36`
- Goal: Generate a full PLY sequence after applying the first manual placement candidate estimated from SuperSplat so the adjusted dynamic asset can be rechecked in the viewer.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/export_dreamscene4d_ply_sequence.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/data/raw_video/forest_walk_0806_0810/manual_place_dynamic_supersplat_try1/manual_transform.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d.pkl;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/gaussians/gaussians/forest_walk_right_vipepose_quick_1_4d_global_motion.pkl`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_manual_try1/sequence_meta.json;/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_manual_try1/forest_walk_right_vipepose_manual_try1_000000.ply`
- Note(s):
  - This first sequence export applies the manual translation directly and approximates the viewer-reported -36 degree z rotation as a world-space z-axis rotation for a quick directional check. If this heads in the right direction, the next step is to compose the manual delta with the existing base rotation rigorously.

### Prepare second manual transform from SplatTransform UI values

- Time: `2026-03-23 06:47:54`
- Goal: Reapply a new manual transform measured in the viewer to the exported DreamScene4D PLY sequence for another qualitative placement test.
- Script(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/scripts/export_dreamscene4d_ply_sequence.py`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_manual_try1/forest_walk_right_vipepose_manual_try1_000000.ply;/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_lama_dilate7_gs_7k_v2/point_cloud/iteration_7000/point_cloud.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_manual_try2`
- Note(s):
  - Target transform from the UI screenshot: position (10.769, -18.052, 10.768), rotation (-35.6, 20.9, -29.14), scale 1.0.

### Test installed splat-transform CLI

- Time: `2026-03-23 06:53:37`
- Goal: Verify the locally installed splat-transform executable and, if available, use it to apply a manual transform to the exported DreamScene4D PLY sequence.
- Script(s):
  - `splat-transform`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_autoalign/forest_walk_right_vipepose_000000.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_splat_transform_try`
- Note(s):
  - Checking the actual installed CLI name and argument schema before applying the viewer-reported transform through splat-transform itself.

### Apply splat-transform to aligned PLY sequence

- Time: `2026-03-23 06:54:11`
- Goal: Use the installed splat-transform CLI to reapply a viewer-measured manual transform directly to the exported DreamScene4D PLY sequence.
- Script(s):
  - `splat-transform`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_autoalign`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_splat_transform_try`
- Note(s):
  - Applied translation (10.769,-18.052,10.768), rotation (-35.6,20.9,-29.14), scale 1.0 to all 20 PLY frames using the installed splat-transform v1.9.2 CLI.

### Check CloudCompare ICP prerequisites for tmp point cloud

- Time: `2026-03-23 06:56:02`
- Goal: Verify that the tmp point cloud exists and determine whether a CloudCompare CLI is available before attempting an ICP registration workflow.
- Script(s):
  - `CloudCompare`
- Input(s):
  - `/mnt/d/develop/master_thesis/tmp/point_cloud.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/tmp/point_cloud.ply`
- Note(s):
  - Checking only the presence of the requested point cloud and CloudCompare executable before starting ICP.

### Prepare registration of transformed PLY sequence to static GS point cloud

- Time: `2026-03-23 07:03:34`
- Goal: Register the manually transformed DreamScene4D PLY sequence against the static forest_walk_right_vipe_gs_7k point cloud by estimating a single rigid/similarity transform and applying it to all frames.
- Script(s):
  - `open3d_icp_or_conversion`
- Input(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_splat_transform_try;/mnt/d/develop/master_thesis/DynamicPoint/output/forest_walk_right_vipe_gs_7k/point_cloud/iteration_7000/point_cloud.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene4d_forest_walk_right_vipepose_quick/ply_sequence_splat_transform_icp`
- Note(s):
  - Checking static point cloud schema and available local ICP libraries before converting the GS model into an ICP-friendly XYZ/RGB point cloud and solving a sequence-wide alignment.

### Clone UniSH repository for review

- Time: `2026-03-30 00:10:56`
- Goal: Fetch the UniSH repository locally and inspect whether it contains a usable alignment-related network or pipeline for scene/object registration.
- Script(s):
  - `git`
- Command(s):
```bash
git clone https://github.com/murphylmf/UniSH.git /mnt/d/develop/master_thesis/external/UniSH
```
- Input(s):
  - `https://github.com/murphylmf/UniSH`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/UniSH`
- Note(s):
  - Repository cloned for code review and method assessment before deciding whether it fits the current DynamicPoint alignment workflow.

### Inspect UniSH environment requirements

- Time: `2026-03-30 00:49:31`
- Goal: Review UniSH environment and build scripts before creating a dedicated conda environment and compiling dependencies.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/environment.yml;/mnt/d/develop/master_thesis/external/UniSH/install.sh`
- Input(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/environment.yml;/mnt/d/develop/master_thesis/external/UniSH/install.sh`
- Output(s):
  - `/root/miniconda3/envs/unish`
- Note(s):
  - Environment inspection precedes installation because UniSH compiles heavy dependencies including PyTorch3D, MMCV, and SAM2.

### Create UniSH environment and start dependency build

- Time: `2026-03-30 00:50:48`
- Goal: Set up a dedicated UniSH conda environment and compile the project dependencies required for inference on this machine.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/install.sh`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate unish && cd /mnt/d/develop/master_thesis/external/UniSH && bash install.sh 12.1
```
- Input(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/environment.yml;/mnt/d/develop/master_thesis/external/UniSH/install.sh`
- Output(s):
  - `/root/miniconda3/envs/unish;/mnt/d/develop/master_thesis/external/UniSH`
- Note(s):
  - Using CUDA 12.1 path on an RTX 4070 Laptop GPU because the local driver reports CUDA 12.x and the UniSH installer maps 12.x to the cu121 wheel index.

### UniSH install blocked by disk space

- Time: `2026-03-30 00:56:14`
- Goal: Diagnose the failed UniSH installation and identify the immediate storage bottleneck before retrying the dependency build.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/install.sh`
- Command(s):
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate unish && cd /mnt/d/develop/master_thesis/external/UniSH && bash install.sh 12.1
```
- Input(s):
  - `/root/miniconda3/envs/unish;/mnt/d/develop/master_thesis/external/UniSH`
- Output(s):
  - `/root/miniconda3/envs/unish`
- Note(s):
  - The install stopped during the PyTorch wheel installation phase with OSError [Errno 28] No space left on device, so storage usage must be reduced before retrying.

### Run Scene360 case1 preprocessing

- Time: `2026-04-28 04:14:19`
- Goal: Run scripts/preprocess_pipeline.sh end-to-end on Scene360 data/case1/frames with RUN_COMPARE=1, producing SAM3 masks, pinhole views/camera poses, LaMa outputs, and inpainted frames.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && RUN_COMPARE=1 bash scripts/preprocess_pipeline.sh data/case1/frames
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/frames`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/frames_preprocess_5fps_full_0428034710`
- Note(s):
  - Completed with exit code 0 on 2026-04-28. Input had 69 PNG frames. Verified outputs: detect/masks 69 files, detect/overlays 69 files, pinhole/views 69 files, pinhole/masks 69 files, cam_pose.json present (17845 bytes), lama_output 162 PNG outputs plus metadata, frames_inpainted 69 PNG frames plus meta.json. SAM3 warned about frame filenames not matching numeric-only format and used lexicographic sort; this is OK for zero-padded frame_000000.png names.

### Run Scene360 case1 first-frame inpaint with 8px mask dilation

- Time: `2026-04-28 04:26:24`
- Goal: Add the preprocess_pipeline.sh LAMA_MASK_DILATE_PIXELS parameter and rerun Scene360 preprocessing on only frame_000000.png with 8 redundant inpaint mask pixels.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && rm -rf runs/case1_first_frame_input && mkdir -p runs/case1_first_frame_input && cp data/case1/frames/frame_000000.png runs/case1_first_frame_input/frame_000000.png && RUN_COMPARE=1 LAMA_MASK_DILATE_PIXELS=8 RUN_ROOT=/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8 bash scripts/preprocess_pipeline.sh runs/case1_first_frame_input
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/frames/frame_000000.png;/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_input`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8`
- Note(s):
  - Completed with exit code 0 on 2026-04-28. Verified lama_input/meta.json has dilate_pixels=8, frames_total=1, patches_total=3, frame_id=000000. Outputs include 1 detect mask, 1 pinhole view, cam_pose.json, 3 LaMa ROI PNG outputs plus metadata, and 1 assembled inpainted frame plus meta.json.

### Export Scene360 SAM3 masks by object

- Time: `2026-04-28 04:32:50`
- Goal: Update SAM3 detection post-processing to emit per-object mask directories and backfill object_masks directories from existing raw SAM3 masks for the case1 first-frame and 69-frame runs.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scene360/preprocess/detect_dynamic_sam3.py`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && python3 - <<'PY' ... _merge_raw_masks(raw_dir=detect/raw, frame_paths=collect_frame_paths(...), masks_dir=detect/masks, overlays_dir=detect/overlays, dilate_pixels=0) ... PY
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/raw;/mnt/d/develop/master_thesis/Scene360/runs/frames_preprocess_5fps_full_0428034710/preprocess/sam3/detect/raw`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/object_masks;/mnt/d/develop/master_thesis/Scene360/runs/frames_preprocess_5fps_full_0428034710/preprocess/sam3/detect/object_masks`
- Note(s):
  - Per-object directories obj000 through obj003 were generated. The first-frame run has 1 mask per object. The 69-frame run has 69 masks per object. Existing detect/masks/frame_*_obj000.png remains the union mask used by default pinhole/inpaint.

### Run Scene360 case1 SAM3-only preprocessing

- Time: `2026-04-28 04:43:30`
- Goal: Run scripts/preprocess_pipeline.sh in SAM3-only mode on data/case1/frames, generating raw masks, union masks, overlays, and per-object mask directories without pinhole or inpainting.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && RUN_SAM3_ONLY=1 RUN_ROOT=/mnt/d/develop/master_thesis/Scene360/runs/case1_sam3_only_0428 bash scripts/preprocess_pipeline.sh data/case1/frames
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/frames`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_sam3_only_0428/preprocess/sam3/detect`
- Note(s):
  - Completed with exit code 0 on 2026-04-28. Verified raw masks=276, union masks=69, overlays=69, object_masks/obj000..obj003 each has 69 masks. Confirmed no lama_input, lama_output, frames_inpainted, or pinhole directories were created.

### Refactor Scene360 preprocessing script into controllable steps

- Time: `2026-04-28 04:55:04`
- Goal: Make scripts/preprocess_pipeline.sh control each preprocessing step independently: frame extraction, masks, pinhole, cam_pose, LaMa input, LaMa, and inpaint assembly.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Command(s):
```bash
bash -n scripts/preprocess_pipeline.sh && RUN_LAMA_INPUT=1 RUN_ROOT=/mnt/d/develop/master_thesis/Scene360/runs/script_step_control_smoke LAMA_MASKS_DIR=/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/masks bash scripts/preprocess_pipeline.sh runs/case1_first_frame_input && RUN_PINHOLE=1 RUN_ROOT=/mnt/d/develop/master_thesis/Scene360/runs/script_step_control_pinhole_smoke PINHOLE_MASKS_DIR=/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/object_masks/obj001 bash scripts/preprocess_pipeline.sh runs/case1_first_frame_input
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_input;/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/masks;/mnt/d/develop/master_thesis/Scene360/runs/case1_first_frame_inpaint_dilate8/preprocess/sam3/detect/object_masks/obj001`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/script_step_control_smoke;/mnt/d/develop/master_thesis/Scene360/runs/script_step_control_pinhole_smoke`
- Note(s):
  - Added step switches RUN_MASKS, RUN_PINHOLE, RUN_CAM_POSE, RUN_LAMA_INPUT, RUN_LAMA, RUN_ASSEMBLE_INPAINT, RUN_INPAINT, plus mask override paths PINHOLE_MASKS_DIR and LAMA_MASKS_DIR. Verified syntax with bash -n, generated LaMa input only (1 frame, 3 patches), and generated pinhole plus cam_pose from object_masks/obj001 without running SAM3 or inpainting.

### Document Scene360 preprocessing script with bilingual comments

- Time: `2026-04-28 04:56:27`
- Goal: Add Chinese and English comments to scripts/preprocess_pipeline.sh explaining each step switch and the behavior of frame extraction, mask generation, pinhole export, cam_pose conversion, LaMa input, LaMa inpainting, and assembly.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && bash -n scripts/preprocess_pipeline.sh
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Note(s):
  - Documentation-only script update. bash -n passed. No preprocessing data was regenerated.

### Add Scene360 preprocessing README

- Time: `2026-04-28 05:06:47`
- Goal: Document the step-controlled Scene360 preprocessing workflow, including masks, per-object masks, pinhole/cam_pose, and inpainting usage examples.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/docs/preprocessing_readme.md`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && sed -n '1,260p' docs/preprocessing_readme.md && wc -l docs/preprocessing_readme.md
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/preprocess_pipeline.sh`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/docs/preprocessing_readme.md`
- Note(s):
  - Documentation-only addition. README has 317 lines and covers RUN_MASKS, RUN_PINHOLE, RUN_CAM_POSE, RUN_LAMA_INPUT, RUN_LAMA, RUN_ASSEMBLE_INPAINT, RUN_INPAINT, RUN_COMPARE, RUN_SAM3_ONLY, and object_masks/objXXX workflows.

### Create Scene360 case-based PLY asset layout

- Time: `2026-04-28 05:23:35`
- Goal: Add a reusable assets directory convention for case-scoped static PLY files and per-object dynamic PLY sequences.
- Script(s):
  - `/usr/bin/mkdir; Codex apply_patch`
- Command(s):
```bash
mkdir -p /mnt/d/develop/master_thesis/Scene360/assets/_template/static /mnt/d/develop/master_thesis/Scene360/assets/_template/dynamic/object_001; apply_patch added assets/README.md and .gitkeep placeholders
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/assets`
- Note(s):
  - Created an empty _template case only; no existing PLY assets were moved or copied.

### Add case1 PLY asset scaffold

- Time: `2026-04-28 05:24:30`
- Goal: Create the concrete case1 asset directories with one static slot and dynamic per-object PLY sequence slots for obj000 through obj003.
- Script(s):
  - `/usr/bin/mkdir; Codex apply_patch`
- Command(s):
```bash
mkdir -p /mnt/d/develop/master_thesis/Scene360/assets/case1/static /mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj000 /mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj001 /mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj002 /mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj003; apply_patch added .gitkeep placeholders
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1;/mnt/d/develop/master_thesis/Scene360/runs/case1_sam3_only_0428/preprocess/sam3/detect/object_masks`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/assets/case1`
- Note(s):
  - Created directory placeholders only. Static scene.ply and dynamic frame_*.ply sequences should be copied in after reconstruction/export.

### Populate case1 static and obj0 dynamic PLY assets

- Time: `2026-04-28 05:27:41`
- Goal: Copy the final static Gaussian PLY, COLMAP sparse metadata, and obj0 dynamic PLY sequence into the case-based Scene360 assets directory.
- Script(s):
  - `/usr/bin/cp; /usr/bin/mkdir; Codex apply_patch`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && mkdir -p assets/case1/static/sparse assets/case1/dynamic/obj0 && cp -a runs/my_static_scene_full_g4d/model/point_cloud/iteration_10000/point_cloud.ply assets/case1/static/scene.ply && cp -a runs/my_static_scene_full_g4d/source/sparse/0 assets/case1/static/sparse/0 && cp -a runs/girl_ply_seq_sam3/*.ply assets/case1/dynamic/obj0/ && cp -a runs/girl_ply_seq_sam3/export_dynamic_ply_summary.json assets/case1/dynamic/obj0/
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/model/point_cloud/iteration_10000/point_cloud.ply;/mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/source/sparse/0;/mnt/d/develop/master_thesis/Scene360/runs/girl_ply_seq_sam3`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/assets/case1/static;/mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj0`
- Note(s):
  - Copied assets, did not move source run outputs. Verified dynamic obj0 contains 27 PLY files and static contains scene.ply plus sparse/0 cameras.txt and images.txt. Removed earlier empty obj000-obj003 placeholders to match the requested obj0 layout.

### Replace case1 obj0 assets with calibrated static-world PLY sequence

- Time: `2026-04-28 05:30:22`
- Goal: Correct assets/case1/dynamic/obj0 to use the calibrated girl PLY sequence instead of the raw uncalibrated export.
- Script(s):
  - `/usr/bin/rm; /usr/bin/cp`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360 && rm -f assets/case1/dynamic/obj0/*.ply assets/case1/dynamic/obj0/export_dynamic_ply_summary.json assets/case1/dynamic/obj0/batch_pose_summary.json && cp -a runs/girl_ply_seq_sam3_rotx_y135_staticworld/*.ply assets/case1/dynamic/obj0/ && cp -a runs/girl_ply_seq_sam3_rotx_y135_staticworld/batch_pose_summary.json assets/case1/dynamic/obj0/
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/girl_ply_seq_sam3_rotx_y135_staticworld;/mnt/d/develop/master_thesis/Scene360/runs/girl_ply_seq_sam3`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/assets/case1/dynamic/obj0`
- Note(s):
  - The raw runs/girl_ply_seq_sam3 export is uncalibrated. The selected calibrated sequence records input_dir=runs/girl_ply_seq_sam3, output_dir=runs/girl_ply_seq_sam3_rotx_y135_staticworld, placement_summary=object_into_static_world_try_axes/placement_summary.json, rotation_summary=object_into_static_world_try_axes_from_rotx_y90_y135/rotation_variants_summary.json, base_axis_key=x, rotation_key=y:135.0, and coarse_scale_final=0.303163604929209. Verified 27 PLY files in assets/case1/dynamic/obj0.

### Generate SAM3 masks for Scene360 case1 views

- Time: `2026-04-29 03:48:30`
- Goal: Run SAM3 tracking segmentation on Scene360 data/case1/views using text prompt 'a person' and export raw masks, merged per-frame masks, overlays, and metadata.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/run_scene360.py`
- Command(s):
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/Scene360/run_scene360.py detect_sam3 --frames /mnt/d/develop/master_thesis/Scene360/data/case1/views --outdir /mnt/d/develop/master_thesis/Scene360/data/case1/sam3 --prompt 'a person' --python-bin /root/miniconda3/envs/sam3/bin/python --sam3-script /mnt/d/develop/master_thesis/Scene360/scene360/preprocess/run_sam3_trackseg.py --start-frame 0 --direction forward --save-all-objects
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/views`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/sam3`
- Note(s):
  - Input contains 40 lexicographically sorted PNG frames named frame_000000.png ... frame_000039.png. Output summary reports 40/40 frames with masks; raw contains 54 object masks because SAM3 emitted a second object on later frames, while merged masks contains one union mask per frame.

### Run 360MonoDepth for Scene360 case1 ERP frame 000000

- Time: `2026-04-29 07:10:40`
- Goal: Generate 360MonoDepth depth for the case1 frame corresponding to the viewed image.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/run_360monodepth_export_npy.sh`
- Command(s):
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so INPUT_RGB=/mnt/d/develop/master_thesis/Scene360/data/case1/frames/frame_000000.png EXPNAME=case1_erp_frame000000_360monodepth_midas2_mean PYTHON_BIN=/root/miniconda3/envs/sam3/bin/python PERSP_MONODEPTH=midas2 BLENDING_METHOD=mean bash /mnt/d/develop/master_thesis/Scene360/scripts/run_360monodepth_export_npy.sh
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/frames/frame_000000.png`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/third_party/360monodepth/results/case1_erp_frame000000_360monodepth_midas2_mean`
- Note(s):
  - The displayed view image /mnt/d/develop/master_thesis/Scene360/data/case1/views/frame_000000.png is 1048x1472 and was rejected by 360MonoDepth because it is not a 2:1 ERP panorama. The successful run used the corresponding 2048x1024 ERP frame. sam3 Python needed LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so for EigenSolvers; Ceres reached the max iteration limit but produced 2048x1024 finite float32 depth and preview PNG.

### Run 360MonoDepth for Scene360 case1 static input

- Time: `2026-04-29 07:21:41`
- Goal: Generate 360MonoDepth depth for data/case1/static/input.png.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/run_360monodepth_export_npy.sh`
- Command(s):
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so INPUT_RGB=/mnt/d/develop/master_thesis/Scene360/data/case1/static/input.png EXPNAME=case1_static_input_360monodepth_midas2_mean PYTHON_BIN=/root/miniconda3/envs/sam3/bin/python PERSP_MONODEPTH=midas2 BLENDING_METHOD=mean bash /mnt/d/develop/master_thesis/Scene360/scripts/run_360monodepth_export_npy.sh
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/data/case1/static/input.png`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/third_party/360monodepth/results/case1_static_input_360monodepth_midas2_mean`
- Note(s):
  - Input is 2048x1024 ERP. sam3 Python needed LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so for EigenSolvers. Ceres reached the max iteration limit but produced 2048x1024 finite float32 depth and preview PNG.

### Run SAM3 ground/floor prompts on Scene360 panorama

- Time: `2026-05-03 23:36:42`
- Goal: Compare SAM3 text prompts ground and floor on runs/my_static_scene_full_g4d/source/panorama.png and export masks, overlays, and a comparison image.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/run_scene360.py`
- Command(s):
```bash
/root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/Scene360/run_scene360.py detect_sam3 --frames /mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/sam3_prompt_ground_floor/input_frames --outdir /mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/sam3_prompt_ground_floor/ground --prompt ground --python-bin /root/miniconda3/envs/sam3/bin/python --sam3-script /mnt/d/develop/master_thesis/Scene360/scene360/preprocess/run_sam3_trackseg.py --start-frame 0 --direction forward --max-frames 1 --save-all-objects && /root/miniconda3/envs/sam3/bin/python /mnt/d/develop/master_thesis/Scene360/run_scene360.py detect_sam3 --frames /mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/sam3_prompt_ground_floor/input_frames --outdir /mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/sam3_prompt_ground_floor/floor --prompt floor --python-bin /root/miniconda3/envs/sam3/bin/python --sam3-script /mnt/d/develop/master_thesis/Scene360/scene360/preprocess/run_sam3_trackseg.py --start-frame 0 --direction forward --max-frames 1 --save-all-objects
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/source/panorama.png`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/my_static_scene_full_g4d/sam3_prompt_ground_floor`
- Note(s):
  - Input was symlinked as one frame named frame_000000.png. SAM3 used local checkpoint /mnt/d/develop/master_thesis/.hf_cache/sam3/sam3.pt. Prompt ground produced one mask covering 244567 pixels / 29.8544 percent of the 1280x640 image; prompt floor produced no mask.

### Visualize Scene360 case1 object rays as Gaussian splats

- Time: `2026-05-10 21:35:40`
- Goal: Generate WebXR-checkable Gaussian PLY markers for camera.json OpenCV rays and cam_pose rays to debug object placement direction.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/visualize_camera_rays_as_gaussians.py`
- Command(s):
```bash
python3 scripts/visualize_camera_rays_as_gaussians.py --depth-dir runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --cameras-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json --cam-pose-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json --outdir PanoScene4D/webxr_viewer/scenes/case1_camera_rays_gs/assets --source cameras-json-opencv --frame-stride 2 --depth-stat median --depth-max 9999 ; python3 scripts/visualize_camera_rays_as_gaussians.py --depth-dir runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --cameras-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json --cam-pose-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json --outdir PanoScene4D/webxr_viewer/scenes/case1_camera_rays_gs/assets --source cam-pose --frame-stride 2 --depth-stat median --depth-max 9999
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_camera_rays_gs/assets`
- Note(s):
  - The generated rays are debug splats only: cyan/yellow uses cameras.json OpenCV R.T and magenta/orange uses cam_pose. Frame stride is 2, so 46 frames are visualized.

### Create Scene360 WebXR background plus dynamic object scene

- Time: `2026-05-11 03:51:54`
- Goal: Load case1 cube6 SHARP/DA360 background once and play the refined face-ray-flip-y object sequence on top for floating/contact inspection.
- Script(s):
  - `/bin/bash`
- Command(s):
```bash
mkdir -p PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_flip_y_seq/assets; ln -s background result.ply and 91 placed object PLY frames; copy sequence.json; add open_url.txt
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_cube6_sharp360_da360_20260511_021834/result.ply; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/placed_from_refined_da360_opencvR_face_ray_flip_y`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_flip_y_seq`
- Note(s):
  - Background is symlinked and loaded via URL load= once; dynamic object is loaded through sequence=. This avoids duplicating the 266M background into each of the 91 object frames.

### Create Scene360 background contact-aligned object sequence

- Time: `2026-05-11 04:05:24`
- Goal: Fix apparent floating by translating the already-correct face-ray object sequence along world Y to match nearby cube6 background contact surface.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/contact_align_placed_ply_to_background.py`
- Command(s):
```bash
python3 scripts/contact_align_placed_ply_to_background.py --ply-dir runs/case1_stride2_030_210_sam3/placed_from_refined_da360_opencvR_face_ray --background-ply runs/case1_cube6_sharp360_da360_20260511_021834/result.ply --outdir runs/case1_stride2_030_210_sam3/placed_from_refined_da360_opencvR_face_ray_bg_contact_y --background-y-percentile 75 --object-y-percentile 99 --search-margin-m 0.25 --max-shift-m 2.0
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/placed_from_refined_da360_opencvR_face_ray; /mnt/d/develop/master_thesis/Scene360/runs/case1_cube6_sharp360_da360_20260511_021834/result.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/placed_from_refined_da360_opencvR_face_ray_bg_contact_y; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_bg_contact_y_seq`
- Note(s):
  - The previous 2D bbox-bottom alignment was rejected because it over-shifted many frames. This contact version uses nearby background XZ points and aligns object Y p99 to background Y p75; mean Y shift is about +1.12m.

### Visualize Scene360 mask-bottom contact rays

- Time: `2026-05-11 04:14:16`
- Goal: Generate Gaussian ray markers from each mask bottom/contact pixel using refined DA360 depth and create a WebXR scene with background, object sequence, and bottom rays.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/visualize_camera_rays_as_gaussians.py`
- Command(s):
```bash
python3 scripts/visualize_camera_rays_as_gaussians.py --depth-dir runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360 --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --cameras-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json --cam-pose-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json --outdir runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_bottom --source cameras-json-opencv --anchor-mode mask-bottom --bottom-band-px 8 --frame-stride 2 --depth-stat median --depth-max 9999
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks; /mnt/d/develop/master_thesis/Scene360/runs/case1_cube6_sharp360_da360_20260511_021834/result.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_bottom; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_with_mask_bottom_rays`
- Note(s):
  - Mask-bottom anchor uses the bottom 8 mask rows and median refined depth; output contains 46 stride-2 camera-json OpenCV rays.

### Mark Scene360 mask bottom points on pinhole masks

- Time: `2026-05-11 04:21:16`
- Goal: Draw the image-space mask-bottom/contact point used for bottom rays directly on each mask/view overlay.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/mark_mask_bottom_points.py`
- Command(s):
```bash
python3 scripts/mark_mask_bottom_points.py --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --image-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/views --outdir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/mask_bottom_points --bottom-band-px 8 --marker-radius 8
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/views`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/mask_bottom_points`
- Note(s):
  - Bottom point is image-space y-down mask bottom: center of bottom 8 mask rows. This is separate from world-space Y-up contact reasoning.

### Visualize Scene360 mask extreme rays

- Time: `2026-05-11 04:25:49`
- Goal: Generate four Gaussian ray PLYs for mask bottom, top, left, and right anchors using refined DA360 depth, with distinct colors and a combined WebXR inspection scene.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/visualize_camera_rays_as_gaussians.py`
- Command(s):
```bash
for mode in mask-bottom mask-top mask-left mask-right; do python3 scripts/visualize_camera_rays_as_gaussians.py --depth-dir runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360 --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --cameras-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json --cam-pose-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json --outdir runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_extremes --source cameras-json-opencv --anchor-mode  --extreme-band-px 8 --frame-stride 2 --depth-stat median --depth-max 9999; done
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_extremes; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_with_mask_extreme_rays`
- Note(s):
  - Each PLY contains 46 stride-2 frames; colors are bottom yellow/cyan, top green, left magenta, right orange, with white camera origins.

### Visualize Scene360 mask extremes in Y-down world coordinates

- Time: `2026-05-11 05:05:37`
- Goal: Regenerate mask top/bottom/left/right rays using the actual Y-down 3D world semantics: top is lower world Y and bottom is higher world Y.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/visualize_camera_rays_as_gaussians.py`
- Command(s):
```bash
for mode in mask-ydown-bottom mask-ydown-top mask-ydown-left mask-ydown-right; do python3 scripts/visualize_camera_rays_as_gaussians.py --depth-dir runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360 --mask-dir runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks --cameras-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cameras.json --cam-pose-json runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/cam_pose.json --outdir runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_ydown_extremes --source cameras-json-opencv --anchor-mode  --extreme-band-px 8 --frame-stride 2 --depth-stat median --depth-max 9999; done
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/ml_depth_pro/pinhole_obj002/views_refined_da360; /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002/raw_masks`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/camera_rays_refined_da360_opencvR_mask_ydown_extremes; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_cube6_bg_obj002_face_ray_with_mask_ydown_extreme_rays`
- Note(s):
  - This supersedes the Y-up mask-world extreme scene for this dataset. Verified 46/46 frames have topY < bottomY under Y-down semantics.

### Stage case1 0511 static background and dynamic PLY sequence for WebXR

- Time: `2026-05-11 06:51:22`
- Goal: Copy the DA360 static background and refined flip-y dynamic object sequence into the PanoScene4D WebXR scene directory, renaming dynamic frames to the viewer's sequential frame_000000.ply convention and generating sequence.json/open_url.txt.
- Script(s):
  - `/bin/bash; /usr/bin/python3`
- Command(s):
```bash
mkdir -p /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511/assets; cp .../case1_cube6_sharp360_da360_20260511_021834/result.ply assets/background_static_da360.ply; for frame_*_masked_noblack.ply copy as frame_%06d.ply; python3 generate sequence.json and open_url.txt
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/ml_sharp_pinhole_obj002_sam3_obj000_all/placed_from_depth_world_refined_da360_flip_y; /mnt/d/develop/master_thesis/Scene360/runs/case1_cube6_sharp360_da360_20260511_021834/result.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511`
- Note(s):
  - Generated 91 dynamic frames named assets/frame_000000.ply through assets/frame_000090.ply; static background is assets/background_static_da360.ply; sequence name is case1_0511_dynamic_obj002_refined_da360_flip_y.

### Start PanoScene4D WebXR preview server for case1_0511

- Time: `2026-05-11 06:51:51`
- Goal: Serve the WebXR viewer locally for the staged case1_0511 static-plus-dynamic scene.
- Script(s):
  - `/usr/bin/python3`
- Command(s):
```bash
cd /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer && python3 -m http.server 3000
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511`
- Output(s):
  - `http://127.0.0.1:3000/index.html?load=/scenes/case1_0511/assets/background_static_da360.ply&filename=case1_0511_static_background_da360.ply&sequence=/scenes/case1_0511/sequence.json`
- Note(s):
  - Verified index.html, sequence.json, and background_static_da360.ply return HTTP 200 from the local server.

### Replace case1_0511 WebXR dynamic sequence with grounded frame0 vertical ydown version

- Time: `2026-05-11 07:55:26`
- Goal: Update the PanoScene4D WebXR case1_0511 scene to use the grounded_by_frame0_vertical_ydown dynamic PLY sequence while keeping the existing DA360 static background.
- Script(s):
  - `/bin/bash; /usr/bin/python3`
- Command(s):
```bash
find /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511/assets -maxdepth 1 -type f -name 'frame_*.ply' -delete; copy frame_*_masked_noblack.ply from placed_from_depth_world_refined_da360_flip_y_grounded_by_frame0_vertical_ydown as assets/frame_%06d.ply; regenerate sequence.json/open_url.txt
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/ml_sharp_pinhole_obj002_sam3_obj000_all/placed_from_depth_world_refined_da360_flip_y_grounded_by_frame0_vertical_ydown`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511`
- Note(s):
  - Replaced 91 dynamic frames; static background remained assets/background_static_da360.ply; verified local server on port 3000 returns the updated sequence manifest.

### Convert PanoScene4D case1_0511 scene to SOG for WebXR

- Time: `2026-05-11 21:17:42`
- Goal: Create a SOG version of the staged case1_0511 WebXR scene and update the viewer so frame_*.sog manifests are treated as playable dynamic sequences.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/viewer_supersplat/node_modules/.bin/splat-transform; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js`
- Command(s):
```bash
splat-transform -q -w -g cpu case1_0511/assets/background_static_da360.ply case1_0511_sog/assets/background_static_da360.sog; for frame_*.ply run splat-transform -q -w -g cpu frame_NNNNNN.ply frame_NNNNNN.sog; patch sequence regex to accept .sog/.splat/.ksplat/.spz and regenerate sequence.json/open_url.txt
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog`
- Note(s):
  - Generated 91 dynamic SOG frames plus background_static_da360.sog; scene size is about 145MB versus about 792MB for the PLY scene; verified HTTP 200 for SOG background and updated manifest on port 3000.

### Create near-field decimated SOG backgrounds for case1_0511 WebXR

- Time: `2026-05-11 21:46:02`
- Goal: Reduce VR load by deleting far static background Gaussians and producing lighter SOG background variants.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/viewer_supersplat/node_modules/.bin/splat-transform`
- Command(s):
```bash
splat-transform -q -w -g cpu background_static_da360.ply -S 0,0,0,30 background_static_da360_near_r30.sog; splat-transform -q -w -g cpu background_static_da360.ply -S 0,0,0,30 -F 25% background_static_da360_near_r30_dec25.sog
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511/assets/background_static_da360.ply`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog/assets/background_static_da360_near_r30.sog; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog/assets/background_static_da360_near_r30_dec25.sog`
- Note(s):
  - Radius 30 around origin retained about 2.74M of 4.97M splats before decimation. near_r30 is 23MB; near_r30_dec25 is 6.9MB and is the recommended first Quest VR test background.

### Patch PanoScene4D WebXR XR direct splat rendering

- Time: `2026-05-11 22:37:51`
- Goal: Fix VR-only black or single-face splat rendering by switching splats from the SuperSplat editor MRT material to the PlayCanvas native GSplat material during XR direct render, then restoring the editor material after XR exits.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js to add Splat.setXrDirectRender(), call it from Camera.startXrDirectRender()/endXrDirectRender(), keep updateCameraUniforms() active in XR, and bump index.html script cache key to 20260511-xr-native-gsplat; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-xr-native-gsplat
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - Desktop editor rendering still uses the SuperSplat MRT material and custom render passes. XR direct rendering now uses the engine-native GSplat shader path, which is the path expected to work with WebXR stereo framebuffers. Verified JavaScript syntax and HTTP 200 for the cache-busted bundle URL.

### Route PanoScene4D XR splats through World layer

- Time: `2026-05-11 22:44:45`
- Goal: Further isolate the VR-only single-face background issue by rendering splats through PlayCanvas' default World layer in XR direct mode instead of the editor-specific Splat layer, and disabling whole-splat frustum culling during XR.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js to save/restore gsplat layers, set entity.gsplat.layers=[scene.worldLayer.id] and meshInstance.cull=false during Splat.setXrDirectRender(true), restore on exit, bump index.html script cache key to 20260511-xr-world-layer-gsplat; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-xr-world-layer-gsplat
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - This keeps desktop editor rendering unchanged but avoids the custom Splat layer in XR, so WebXR stereo uses the standard World layer render order and view setup. Verified JavaScript syntax and HTTP 200 for the new cache-busted bundle URL.

### Add PanoScene4D XR debug overlay

- Time: `2026-05-11 22:50:33`
- Goal: Expose browser-side XR/session/splat/layer diagnostics for the VR-only single-face rendering issue.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js to add xrDebug=1 overlay, window error/unhandledrejection logging, startXrDirectRender/splat layer diagnostics, and XR session event logs; bump index.html script cache key to 20260511-xr-debug; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-xr-debug
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - Use URL parameter xrDebug=1 to show an in-page overlay and [XRDBG] console logs. This does not change rendering behavior except for logging.

### Preserve WebXR headset pose for GSplat sorting

- Time: `2026-05-11 22:58:26`
- Goal: Fix the VR-only one-face background symptom caused by desktop orbit camera updates overwriting the camera node used by GSplat sorting after WebXR updates the headset pose.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js so Camera.onUpdate() returns early while xrDirectRender/app.xr.active, preserving the WebXR-updated mainCamera pose used by GSplatInstance.sort(camera._node); bump index.html cache key to 20260511-xr-pose-sort; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-xr-pose-sort
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - PlayCanvas XR updates mainCamera from headset pose before app.update(); the viewer's desktop Camera.onUpdate() then overwrote that node before GSplatInstance.update() sorted against camera._node. In XR the view matrices came from XR views but splat order came from the stale desktop orbit camera, which can make a 360 background appear as a single face. Verified syntax and HTTP 200 for the updated bundle.

### Fix PanoScene4D XR background-dynamic draw order

- Time: `2026-05-11 23:26:14`
- Goal: Prevent the static Gaussian background from over-blending or visually covering the dynamic SOG frame sequence in WebXR direct rendering.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js to assign XR drawOrder per splat, switch World layer transparentSortMode to SORTMODE_MANUAL during startXrDirectRender(), restore it on XR exit, and bump index.html cache key to 20260511-xr-dynamic-draworder; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-xr-dynamic-draworder
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/scenes/case1_0511_sog/sequence.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - In XR direct mode all splats are routed through the World layer. The World layer's default transparent sort can order the static background after the dynamic frame because both AABBs are near the origin. During XR this now uses manual transparent sorting: background/static/bg filenames draw first, frame_*/dynamic filenames draw later. Verified syntax and HTTP 200 for the cache-busted bundle.

### Add PanoScene4D realtime FPS overlay

- Time: `2026-05-11 23:42:16`
- Goal: Replace the right-corner in-page XR debug log display with a realtime FPS panel and sparkline graph for WebXR performance checks.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Command(s):
```bash
apply_patch index.js to remove the xrDebugPanel DOM log display, keep xrDebugLog console-only, add initFpsOverlay(app) hooked to PlayCanvas postrender with smoothed FPS and canvas sparkline, call it from Scene setup, and bump index.html cache key to 20260511-fps-overlay; node --check PanoScene4D/webxr_viewer/index.js; curl -I http://127.0.0.1:3000/index.js?v=20260511-fps-overlay
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.js; /mnt/d/develop/master_thesis/Scene360/PanoScene4D/webxr_viewer/index.html`
- Note(s):
  - The FPS overlay is enabled by default and can be hidden with fps=0. It measures actual PlayCanvas postrender cadence, so it tracks XR render ticks rather than a separate browser requestAnimationFrame loop. xrDebug=1 now only writes diagnostics to the browser console.

### Scene360 case1 proper-rotation cam_pose conversion

- Time: `2026-05-12 03:12:28`
- Goal: Generate cam_pose.json and cameras_pose.json from the proper-rotation cameras.json.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/run_scene360.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/Scene360/run_scene360.py convert_pinhole_cameras_to_cam_pose --cameras-json /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cameras.json --out /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cam_pose.json && cp /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cam_pose.json /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cameras_pose.json
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cameras.json`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cam_pose.json,/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot/cameras_pose.json`
- Note(s):
  - cameras_pose.json is a compatibility copy of cam_pose.json because downstream Scene360 uses cam_pose.json naming.

### Scene360 case1 proper-rotation pinhole copy

- Time: `2026-05-12 03:12:28`
- Goal: Create a pinhole_obj002 copy whose cameras.json uses det(R)=+1 proper rotations while preserving the existing pinhole view pixels and masks.
- Script(s):
  - `/mnt/d/develop/master_thesis/Scene360/scripts/make_proper_pinhole_from_existing.py`
- Command(s):
```bash
python3 /mnt/d/develop/master_thesis/Scene360/scripts/make_proper_pinhole_from_existing.py --src /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002 --dst /mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot --copy-sam3
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002`
- Output(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_stride2_030_210_sam3/preprocess/sam3/pinhole_obj002_proper_rot`
- Note(s):
  - Original ERP frame directory recorded in the old summary is absent in this workspace, so this step reuses the existing pinhole views/raw_masks and rewrites cameras.json with R_new = diag(1,-1,1) @ R_old; view pixels are equivalent under the corrected ray convention.

### UniSH Scene360 orientation audit and visualization fix

- Time: `2026-05-12 18:23:07`
- Goal: Verify proper-rotation UniSH PLY orientation numerically and fix visualization-only horizontal mirror sources.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/unish/utils/inference_utils.py; /mnt/d/develop/master_thesis/external/UniSH/visualize_saved_results.py`
- Command(s):
```bash
python3 inline PLY reprojection color check for frames 0,45,90; apply_patch to choose Open3D up by camera determinant and disable default visualize_saved_results horizontal flip; python3 syntax checks; git commit -m 'Fix Scene360 visualization camera orientation'
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_unish_mask_proper_rot_yfixed/views; /mnt/d/develop/master_thesis/external/UniSH`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/UniSH commit 36c6e6e`
- Note(s):
  - PLY reprojection with det(R)=+1 matched u=fx*x/z+cx, v=cy-fy*y/z on frames 0/45/90; horizontal-mirror reprojection had much higher color error. The remaining mirror was caused by visualization fliplr default and old det=-1 up-vector logic.

### UniSH Scene360 proper-camera XY bridge fix

- Time: `2026-05-12 18:34:20`
- Goal: Correct the proper-rotation UniSH world export to match the previously validated views_unmirrored background direction.
- Script(s):
  - `/mnt/d/develop/master_thesis/external/UniSH/unish/utils/inference_utils.py`
- Command(s):
```bash
python3 inline comparison showed old views == proper_rot_yfixed, old views camera-X mirror == views_unmirrored, and proper det=+1 camera with diag(-1,-1,1) bridge exactly matches views_unmirrored; apply_patch changed Scene360 proper camera bridge from Y-only flip to X/Y flip; git commit -m 'Fix Scene360 proper camera XY bridge'
```
- Input(s):
  - `/mnt/d/develop/master_thesis/Scene360/runs/case1_unish_mask/views; /mnt/d/develop/master_thesis/Scene360/runs/case1_unish_mask/views_unmirrored; /mnt/d/develop/master_thesis/Scene360/runs/case1_unish_mask_proper_rot_yfixed/views`
- Output(s):
  - `/mnt/d/develop/master_thesis/external/UniSH commit 5c05cbd`
- Note(s):
  - The new cameras.json is not the source of the horizontal mirror: det(R)=+1 is correct. The bug was the UniSH model-camera to Scene360-camera bridge; it must be diag(-1,-1,1) for proper cameras, not diag(1,-1,1). Existing proper_rot_yfixed outputs can be repaired by one camera-X mirror with mirror_saved_results.py, or regenerated with the fixed code.
