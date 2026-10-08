'''Controlled LiDAR-camera calibration drift benchmark for Topic A.

The alignment metric first associates LiDAR points with each KITTI ground-truth
3D box under the reference calibration. It then perturbs the extrinsic matrix
and measures how many of those same points still land inside the corresponding
labelled 2D box.
'''
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from starter.datasets import list_frames, load_frame
from starter.projection import overlay_points, perturb_extrinsic, project_velo_to_image, velo_to_cam


OBJECT_TYPES = {'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram'}


def points_in_box3d(points_cam: np.ndarray, obj) -> np.ndarray:
    '''Return a mask for points inside a KITTI 3D box in camera coordinates.'''
    h, w, length = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ rotation
    eps = 1e-6
    return (
        (np.abs(local[:, 0]) <= length / 2 + eps)
        & (local[:, 1] <= eps)
        & (local[:, 1] >= -h - eps)
        & (np.abs(local[:, 2]) <= w / 2 + eps)
    )


def indexed_projection(points: np.ndarray, calib, image_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Project points while retaining a pixel/depth slot for every input index.'''
    uv, depth, mask = project_velo_to_image(points, calib, image_shape)
    uv_all = np.full((len(points), 2), np.nan, dtype=float)
    depth_all = np.full(len(points), np.nan, dtype=float)
    uv_all[mask] = uv
    depth_all[mask] = depth
    return uv_all, depth_all, mask


def alignment_counts(uv_all: np.ndarray, valid: np.ndarray, object_masks, objects):
    '''Count associated object points that remain within their labelled 2D box.'''
    totals = {'all': [0, 0], 'near': [0, 0], 'mid': [0, 0], 'far': [0, 0]}
    for obj, object_mask in zip(objects, object_masks):
        indices = np.flatnonzero(object_mask)
        if not len(indices):
            continue
        x1, y1, x2, y2 = obj.bbox
        uv = uv_all[indices]
        hit = (
            valid[indices]
            & (uv[:, 0] >= x1)
            & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1)
            & (uv[:, 1] <= y2)
        )
        distance = float(obj.location[2])
        bucket = 'near' if distance < 20 else ('mid' if distance < 40 else 'far')
        totals['all'][0] += int(hit.sum())
        totals['all'][1] += len(indices)
        totals[bucket][0] += int(hit.sum())
        totals[bucket][1] += len(indices)
    return totals


def safe_pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float('nan')


def evaluate(frame: dict, config: dict) -> dict:
    points = frame['points']
    image_shape = frame['image'].shape
    reference_cam = velo_to_cam(points[:, :3], frame['calib'])
    objects = [obj for obj in frame['labels'] if obj.type in OBJECT_TYPES]
    object_masks = [points_in_box3d(reference_cam, obj) for obj in objects]
    calib = perturb_extrinsic(
        frame['calib'],
        yaw_deg=config['yaw_deg'],
        t_xyz_m=(0.0, config['ty_m'], 0.0),
    )
    uv_all, _, valid = indexed_projection(points, calib, image_shape)
    counts = alignment_counts(uv_all, valid, object_masks, objects)
    row = {
        'frame_id': frame['frame_id'],
        'perturbation': config['name'],
        'level': config['level'],
        'yaw_deg': config['yaw_deg'],
        'ty_m': config['ty_m'],
        'n_points': len(points),
        'inside_fov': int(valid.sum()),
        'inside_fov_pct': safe_pct(int(valid.sum()), len(points)),
        'n_objects': len(objects),
        'object_points': counts['all'][1],
        'aligned_object_points': counts['all'][0],
        'alignment_pct': safe_pct(counts['all'][0], counts['all'][1]),
    }
    for bucket in ('near', 'mid', 'far'):
        row[f'{bucket}_object_points'] = counts[bucket][1]
        row[f'{bucket}_aligned_points'] = counts[bucket][0]
        row[f'{bucket}_alignment_pct'] = safe_pct(counts[bucket][0], counts[bucket][1])
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (name, level, yaw, ty), group in rows.groupby(
        ['perturbation', 'level', 'yaw_deg', 'ty_m'], sort=False
    ):
        out = {
            'perturbation': name,
            'level': level,
            'yaw_deg': yaw,
            'ty_m': ty,
            'n_frames': len(group),
            'inside_fov_pct': 100.0 * group['inside_fov'].sum() / group['n_points'].sum(),
            'alignment_pct': safe_pct(
                int(group['aligned_object_points'].sum()), int(group['object_points'].sum())
            ),
            'object_points': int(group['object_points'].sum()),
        }
        for bucket in ('near', 'mid', 'far'):
            out[f'{bucket}_alignment_pct'] = safe_pct(
                int(group[f'{bucket}_aligned_points'].sum()),
                int(group[f'{bucket}_object_points'].sum()),
            )
            out[f'{bucket}_object_points'] = int(group[f'{bucket}_object_points'].sum())
        summary_rows.append(out)
    return pd.DataFrame(summary_rows)


def save_plot(summary: pd.DataFrame, out_path: Path) -> None:
    baseline = summary.loc[summary['perturbation'] == 'baseline'].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    experiments = [
        ('yaw', 'yaw_deg', 'Yaw error (degrees)'),
        ('translation_y', 'ty_m', 'Lateral translation error (m)'),
    ]
    for ax, (name, x_col, x_label) in zip(axes, experiments):
        subset = pd.concat([
            summary.loc[summary['perturbation'] == 'baseline'],
            summary.loc[summary['perturbation'] == name],
        ]).sort_values(x_col)
        ax.plot(subset[x_col], subset['alignment_pct'], marker='o', linewidth=2, label='3D-to-2D alignment')
        ax.plot(subset[x_col], subset['inside_fov_pct'], marker='s', linewidth=2, label='Points inside FOV')
        ax.axhline(baseline['alignment_pct'], color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Metric (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('Sensitivity of projection metrics to calibration drift')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def visual(frame: dict, yaw_deg: float, title: str) -> tuple[np.ndarray, dict]:
    config = {
        'name': 'baseline' if yaw_deg == 0 else 'yaw',
        'level': yaw_deg,
        'yaw_deg': yaw_deg,
        'ty_m': 0.0,
    }
    calib = perturb_extrinsic(frame['calib'], yaw_deg=yaw_deg)
    uv, depth, _ = project_velo_to_image(frame['points'], calib, frame['image'].shape)
    image = overlay_points(frame['image'], uv, depth, radius=1)
    for obj in frame['labels']:
        if obj.type in OBJECT_TYPES:
            x1, y1, x2, y2 = np.round(obj.bbox).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    metrics = evaluate(frame, config)
    caption = '{}: FOV={:.2f}%  alignment={:.2f}%'.format(
        title, metrics['inside_fov_pct'], metrics['alignment_pct']
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(image, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return image, metrics


def configs() -> list[dict]:
    result = [{'name': 'baseline', 'level': 0.0, 'yaw_deg': 0.0, 'ty_m': 0.0}]
    result.extend(
        {'name': 'yaw', 'level': value, 'yaw_deg': value, 'ty_m': 0.0}
        for value in (0.5, 1.0, 2.0, 3.0)
    )
    result.extend(
        {'name': 'translation_y', 'level': value, 'yaw_deg': 0.0, 'ty_m': value}
        for value in (0.02, 0.05, 0.10)
    )
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the Topic A calibration-drift benchmark')
    parser.add_argument('--data-root', default='data/kitti_mini')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--frames', nargs='*', help='Optional frame IDs; default uses all frames')
    args = parser.parse_args()

    frame_ids = args.frames or list_frames(args.data_root)
    loaded = [load_frame(args.data_root, frame_id) for frame_id in frame_ids]
    rows = [evaluate(frame, config) for frame in loaded for config in configs()]
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_rows(out_dir / 'projection_benchmark.csv', rows)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary.to_csv(out_dir / 'projection_summary.csv', index=False)
    save_plot(summary, figures_dir / 'calibration_drift_metrics.png')

    baseline = detail.loc[detail['perturbation'] == 'baseline']
    best_frame_id = baseline.sort_values('object_points', ascending=False).iloc[0]['frame_id']
    frame = next(item for item in loaded if item['frame_id'] == best_frame_id)
    base_image, base_metrics = visual(frame, 0.0, 'Reference')
    mild_image, _ = visual(frame, 1.0, 'Yaw +1 degree')
    severe_image, severe_metrics = visual(frame, 3.0, 'Yaw +3 degrees')
    cv2.imwrite(str(figures_dir / f'demo_baseline_{best_frame_id}.png'), base_image)
    cv2.imwrite(str(figures_dir / f'demo_yaw_1deg_{best_frame_id}.png'), mild_image)
    cv2.imwrite(str(figures_dir / f'demo_yaw_3deg_{best_frame_id}.png'), severe_image)
    comparison = np.hstack([base_image, severe_image])
    cv2.imwrite(str(figures_dir / f'fail_fov_metric_{best_frame_id}.png'), comparison)

    print(summary.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
    print(f'Representative frame: {best_frame_id}')
    print(
        'Failure evidence: FOV {:.3f}% -> {:.3f}%, alignment {:.3f}% -> {:.3f}%'.format(
            base_metrics['inside_fov_pct'],
            severe_metrics['inside_fov_pct'],
            base_metrics['alignment_pct'],
            severe_metrics['alignment_pct'],
        )
    )
    print(f'Outputs written to {out_dir}')


if __name__ == '__main__':
    main()
