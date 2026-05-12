# UT-X5 GS / Reference Alignment Pipeline

## Scope

This document records the alignment workflow used for:

- `mlsharp` Gaussian Splatting model
- UniSH-exported reference point cloud
- UniSH frame-0 human mask
- final ICP refinement in CloudCompare

The purpose is to keep one reproducible record of:

- which inputs were used
- which scripts were used
- what each stage optimized
- which intermediate outputs are considered the current best results

## Inputs

### GS source

- GS model:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/mlsharp_stage1_calibrated_pose/ut_x5_mlsharp_pose_1_model.ply`

### Reference point cloud

- Cleaned reference cloud:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/unish_human_frame_0000_cleaned/human_frame_0000_cleaned.ply`

### UniSH mask / image / camera

- Input image:
  - `/mnt/d/develop/master_thesis/tmp/utx5-track1-unish-complete/step1_main_view/intermediate/input_frames/frame_0000.png`
- Input mask:
  - `/mnt/d/develop/master_thesis/tmp/utx5-track1-unish-complete/step1_main_view/intermediate/human_masks/mask_0000.png`
- Camera parameters:
  - `/mnt/d/develop/master_thesis/tmp/utx5-track1-unish-complete/step1_main_view/camera_parameters.npz`

## Camera Convention

Both the reference-point-cloud branch and the GS branch were aligned under the same real camera convention:

- same `frame_idx = 0`
- same `w2c` extrinsic from `camera_parameters.npz`
- same native-camera inference from principal point:
  - native size inferred as `518 x 518`
- same resized target resolution:
  - mask/image size `700 x 700`
- same rescaled intrinsics:
  - `fx = fy = 352.1312404323269`
  - `cx = cy = 350.0`

This matters because it makes the final transforms composable in the same external coordinate frame.

## Stage 1: Reference Point Cloud to Mask

Script:

- `/mnt/d/develop/master_thesis/DynamicPoint/scripts/mask_coarse_align_ply.py`

Method:

1. Load the reference point cloud.
2. Project it with the real camera from `camera_parameters.npz`.
3. Optimize a mask-constrained Sim(3) transform.
4. Use true silhouette IoU, not bbox IoU, for the final refinement.
5. Keep rotation frozen and optimize only `scale + translation`.

Why this version was kept:

- allowing rotation made the result less stable
- `s+t only` produced the best silhouette alignment for the reference cloud

Best output:

- directory:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/ref_pointcloud_to_mask_frame0000_st_only_maskiou_095`
- transformed cloud:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/ref_pointcloud_to_mask_frame0000_st_only_maskiou_095/aligned_points.ply`
- matrix:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/ref_pointcloud_to_mask_frame0000_st_only_maskiou_095/estimated_transform.npz`
- summary:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/ref_pointcloud_to_mask_frame0000_st_only_maskiou_095/transform.json`

Best metrics:

- `final_mask_iou = 0.9149062526512259`
- `optimized_scale = 1.1155753644253983`
- `optimized_translation = [-0.0034134886389624117, -0.024782726863736313, 0.24887430939263086]`
- `optimized_rotation = identity`

Final reference transform:

```text
[[ 1.11557536  0.          0.         -0.01341183]
 [ 0.          1.11557536  0.         -0.03593882]
 [ 0.          0.          1.11557536 -0.05788035]
 [ 0.          0.          0.          1.        ]]
```

## Stage 2: GS to Mask

### 2.1 Failed direct real-camera initialization

Script:

- `/mnt/d/develop/master_thesis/DynamicPoint/scripts/gs_mask_align_real_camera.py`

Directly optimizing the GS under the real camera from scratch was too weak:

- best result before improved initialization was only around `0.27` mask IoU

### 2.2 Reused old-camera GS pose, then mapped to the real camera

The stronger version was:

1. First obtain a better GS pose under DreamScene4D's old camera workflow.
2. Convert that pose into a 4x4 Sim(3) transform.
3. Map that transform into the real-camera world using:

```text
T_real_init = inv(W_real) @ W_old @ T_old
```

4. Use this transferred pose as initialization.
5. Run short `mask-only` refinement under the real camera.
6. Save overlay images for every refinement step.

Old-camera GS pose used as initialization:

- `/mnt/d/develop/master_thesis/DynamicPoint/output/dreamscene_main_import_mlsharp_frame0000/gaussians/ut_x5_mlsharp_import_frame0000_calibrated_pose.pkl`

Improved real-camera output:

- directory:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_mask_align_realcam_from_oldpose_frame0000`
- final pose:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_mask_align_realcam_from_oldpose_frame0000/gaussians/ut_x5_mlsharp_realcam_from_oldpose_frame0000_calibrated_pose.pkl`
- summary:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_mask_align_realcam_from_oldpose_frame0000/transform.json`
- final overlay:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_mask_align_realcam_from_oldpose_frame0000/vis/ut_x5_mlsharp_realcam_from_oldpose_frame0000_mask_overlay.png`
- per-step overlays:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_mask_align_realcam_from_oldpose_frame0000/vis/steps`

Best metrics:

- `best_mask_iou = 0.6651731133460999`
- best step at `step_0159`

Final GS transform:

```text
[[ 1.80687887  0.10714585  0.59201933 -0.00264989]
 [-0.18036948  1.8842406   0.20948174  0.00132487]
 [-0.57396339 -0.25482448  1.79789012  1.99999846]
 [ 0.          0.          0.          1.        ]]
```

## Stage 3: Convert Both Branches to 4x4 Matrices

Why this step is necessary:

- both branches save `scale`, `rotation`, `rotation_center`, and `translation`
- these are not directly comparable parameter tuples
- the actual transformation is:

```text
x' = sR(x - c) + c + t
```

Therefore both branches were converted to a common 4x4 matrix representation.

Validation:

- reference params -> matrix exactly matched the saved `estimated_transform.npz`
- GS json params -> matrix exactly matched the GS pose pkl

Matrix check output:

- `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_matrix_check/transforms.json`
- `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_matrix_check/transforms.npz`

Consistency check:

- `ref_diff_max = 0.0`
- `gs_diff_max = 0.0`

## Stage 4: Compose GS to Reference Transform

Once both transforms were in the same external coordinate frame, the relative transform was computed as:

```text
T_gs_to_ref = inv(T_ref) @ T_gs
```

Result:

```text
[[ 1.61968337  0.09604537  0.53068519  0.00964698]
 [-0.16168292  1.6890303   0.1877791   0.03340312]
 [-0.51449988 -0.22842426  1.61162587  1.84467932]
 [ 0.          0.          0.          1.        ]]
```

Outputs:

- `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_matrix_check/transforms.json`
- `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_matrix_check/transforms.npz`

## Stage 5: Apply the Relative Transform Back to the Full GS

The relative matrix was then applied to the full original GS file, not just a display-only point cloud.

Bundle directory:

- `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_overlay_bundle`

Files:

- GS in reference coordinates:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_overlay_bundle/gs_in_ref_coords.ply`
- Reference point cloud in its own coordinates:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_overlay_bundle/reference_pointcloud_in_ref_coords.ply`
- Relative transform:
  - `/mnt/d/develop/master_thesis/DynamicPoint/output/gs_ref_overlay_bundle/gs_to_ref_transform.json`

Purpose:

- these two files are ready to be opened together in CloudCompare
- the GS is still a full GS `.ply`, but CloudCompare will visualize it as a point cloud

## Stage 6: ICP Refinement in CloudCompare

### GUI setup

Loaded clouds:

- data / to be aligned:
  - `gs_in_ref_coords.ply`
- model / reference:
  - `reference_pointcloud_in_ref_coords.ply`

Recommended first-pass ICP settings:

- `Final overlap = 60%`
- `Adjust scale = OFF`
- `Normals = Ignored`
- `RMS difference = 1e-5`

Reason:

- overlap between the GS centers and the reference cloud is partial, not exact 100%
- keeping scale fixed is safer after the previous mask-based scale alignment

### Command-line setup

CloudCompare was installed in WSL via Flatpak:

- package:
  - `org.cloudcompare.CloudCompare`
- version:
  - `2.13.2`

Important runtime notes:

- `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` had been exported as empty strings
- this broke Flatpak networking until they were unset or removed from the command environment

Because Flatpak cannot directly access `/mnt/d/...` by default, the two ICP files were copied to:

- `/root/cloudcompare_icp/gs_in_ref_coords.ply`
- `/root/cloudcompare_icp/reference_pointcloud_in_ref_coords.ply`

Working command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u NO_PROXY QT_QPA_PLATFORM=minimal \
flatpak run org.cloudcompare.CloudCompare \
-SILENT \
-C_EXPORT_FMT PLY \
-PLY_EXPORT_FMT BINARY_LE \
-AUTO_SAVE OFF \
-O "/root/cloudcompare_icp/gs_in_ref_coords.ply" \
-O "/root/cloudcompare_icp/reference_pointcloud_in_ref_coords.ply" \
-ICP \
-OVERLAP 60 \
-MIN_ERROR_DIFF 1e-5 \
-RANDOM_SAMPLING_LIMIT 20000 \
-NO_TIMESTAMP \
-SAVE_CLOUDS FILE "/root/cloudcompare_icp/gs_in_ref_coords_icp.ply /root/cloudcompare_icp/reference_pointcloud_in_ref_coords_icp.ply" \
-LOG_FILE "/root/cloudcompare_icp/cloudcompare_icp.log"
```

Notes:

- `QT_QPA_PLATFORM=minimal` was necessary to avoid the `xcb` display error in the Flatpak environment
- if the command is split across lines, every line continuation must end with `\` and no trailing spaces

## Current Practical Conclusion

The working alignment recipe is:

1. Align the reference point cloud to the real-camera mask with silhouette IoU.
2. Align the GS to the same mask, but initialize from the stronger old-camera GS pose.
3. Convert both branches to 4x4 transforms.
4. Compute `T_gs_to_ref`.
5. Apply that transform back to the full GS.
6. Run ICP between:
   - transformed GS centers
   - reference point cloud

This is better than direct GS-to-pointcloud registration because:

- scale and coarse position are already constrained in 2D by the mask
- GS and reference are first brought into the same camera-conditioned external frame
- ICP is used only as a local geometric refinement step

