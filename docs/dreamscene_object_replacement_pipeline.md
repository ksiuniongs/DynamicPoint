# DreamScene Object Replacement Pipeline

## Purpose

This pipeline treats a `DreamScene360` scene as the static background world and aligns a high-quality object point cloud into that world from a panorama-space mask.

Current scope:

- estimate a coarse absolute object pose in DreamScene world coordinates
- refine the pose against a panorama mask
- export the aligned object point cloud
- optionally prune static background points inside the replacement region
- optionally create preview renders
- optionally merge if the inserted object is also a Gaussian-splat PLY

This is an MVP replacement pipeline. It does not yet use `MASt3R`.

## Entry Script

- [scripts/dreamscene_object_replacement_pipeline.py](/mnt/d/develop/master_thesis/DynamicPoint/scripts/dreamscene_object_replacement_pipeline.py)

## Required Inputs

- A successful DreamScene360 scene output directory
  - example: `/mnt/d/develop/master_thesis/external/DreamScene360/output/gs360_utx5_1024_stride2`
- The corresponding DreamScene360 data directory
  - example: `/mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test`
- One object point cloud `.ply`
- One panorama mask `.png`
  - or a fallback bbox via `--bbox_xyxy`

## What The Script Uses Internally

From DreamScene360:

- `cfg_args` for panorama size and data path
- `data/.../sparse/0/points3D.ply` to reconstruct a panorama-space depth map
- `output/.../point_cloud/iteration_10000/point_cloud.ply` as the static scene for replacement and preview

## Example

### Mask-driven run

```bash
cd /mnt/d/develop/master_thesis/DynamicPoint

/root/miniconda3/envs/sam3/bin/python scripts/dreamscene_object_replacement_pipeline.py \
  --dreamscene_output_dir /mnt/d/develop/master_thesis/external/DreamScene360/output/gs360_utx5_1024_stride2 \
  --dreamscene_data_dir /mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test \
  --panorama_path /mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test/frame_000001.png \
  --mask_png /path/to/object_mask.png \
  --object_ply /path/to/high_quality_object.ply \
  --output_dir /mnt/d/develop/master_thesis/DynamicPoint/output/object_replace_run \
  --prune_static \
  --make_preview
```

### Bbox-only smoke test

```bash
cd /mnt/d/develop/master_thesis/DynamicPoint

/root/miniconda3/envs/sam3/bin/python scripts/dreamscene_object_replacement_pipeline.py \
  --dreamscene_output_dir /mnt/d/develop/master_thesis/external/DreamScene360/output/gs360_utx5_1024_stride2 \
  --dreamscene_data_dir /mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test \
  --panorama_path /mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test/frame_000001.png \
  --bbox_xyxy 390,160,640,360 \
  --object_ply /mnt/d/develop/master_thesis/external/DreamScene360/data/gs360_utx5_1024_test/sparse/0/points3D.ply \
  --object_kind generic \
  --output_dir /mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene_object_replace_smoke \
  --sample_limit 2000 \
  --search_iters 2 \
  --search_candidates 8
```

## Outputs

The pipeline writes these core files:

- `mask_used.png`
- `scene_depth_map.npz`
- `scene_depth_preview.png`
- `aligned_object.ply`
- `aligned_object_depth_map.npz`
- `panorama_replacement_overlay.png`
- `replacement_manifest.json`

If `--prune_static` is enabled:

- `pruned_static_scene.ply`

If `--make_preview` is enabled:

- `preview/novel_view_contact_sheet.png`
- `preview/novel_view_orbit.mp4`

If the object input is detected as `gs` or `--object_kind gs`:

- `aligned_object_transform_for_merge.json`
- `merged_scene_with_object.ply`

## Current Limitations

- No `MASt3R` relative pose yet
- No automatic SAM/SAM3 segmentation inside this script yet
- The object is aligned as a point cloud/GS against a panorama mask, not by multi-view learned correspondence
- Generic object PLY insertion exports aligned overlays and previews, but not a unified Gaussian file

## Existing Helpers It Reuses

- [scripts/merge_registered_gaussians.py](/mnt/d/develop/master_thesis/DynamicPoint/scripts/merge_registered_gaussians.py)
- [scripts/render_registered_novel_views.py](/mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py)

