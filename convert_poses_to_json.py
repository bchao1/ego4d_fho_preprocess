#!/usr/bin/env python3
"""
Convert fused_pose.npy files to camera.json format.

This script converts camera extrinsics from numpy format (4x4 transformation matrices)
to JSON format with quaternion (wxyz) and translation (xyz) representation.

NOTE: This script is to retroactively convert metadata.csv. label_clips.py does this for you now.
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def matrix_to_quaternion_wxyz(rotation_matrix):
    """
    Convert 3x3 rotation matrix to quaternion in wxyz format.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        quaternion: [w, x, y, z] format
    """
    rot = Rotation.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return quat_wxyz


def convert_fused_pose_to_json(fused_pose_path, output_json_path=None):
    """
    Convert fused_pose.npy to camera.json format.
    
    Args:
        fused_pose_path: Path to fused_pose.npy file
        output_json_path: Path to output camera.json file (if None, uses same directory)
        
    Returns:
        Path to created camera.json file
    """
    # Load poses (shape: [N, 4, 4])
    poses = np.load(fused_pose_path)
    
    # Create output path if not specified
    if output_json_path is None:
        output_json_path = os.path.join(os.path.dirname(fused_pose_path), 'camera.json')
    
    # Convert each pose to JSON format
    camera_data = {}
    
    for frame_idx in range(len(poses)):
        c2w = poses[frame_idx]  # Camera-to-world transformation matrix
        
        # Extract translation (xyz)
        translation = c2w[:3, 3].tolist()
        
        # Extract rotation matrix and convert to quaternion (wxyz)
        rotation_matrix = c2w[:3, :3]
        quaternion = matrix_to_quaternion_wxyz(rotation_matrix)
        
        # Store in JSON format (frame indices as strings)
        camera_data[str(frame_idx)] = {
            "translation_xyz": translation,
            "quaternion_wxyz": quaternion
        }
    
    # Write JSON file
    with open(output_json_path, 'w') as f:
        json.dump(camera_data, f, indent=2)
    
    return output_json_path


def process_all_data(data_folder, update_metadata=True):
    """
    Process all fused_pose.npy files in the data folder and create camera.json files.
    
    Args:
        data_folder: Root folder containing video_id/start_frame/ structure
        update_metadata: Whether to update metadata.csv
    """
    data_folder = Path(data_folder)
    metadata_path = data_folder / 'metadata.csv'
    
    # Find all fused_pose.npy files
    fused_pose_files = list(data_folder.rglob('fused_pose.npy'))
    
    print(f"Found {len(fused_pose_files)} fused_pose.npy files")
    
    # Convert each file
    converted_files = []
    failed_files = []
    
    for fused_pose_path in tqdm(fused_pose_files, desc="Converting poses"):
        try:
            output_path = convert_fused_pose_to_json(fused_pose_path)
            converted_files.append((fused_pose_path, output_path))
        except Exception as e:
            print(f"Error converting {fused_pose_path}: {e}")
            failed_files.append(fused_pose_path)
    
    print(f"\nSuccessfully converted {len(converted_files)} files")
    if failed_files:
        print(f"Failed to convert {len(failed_files)} files")
    
    # Update metadata.csv if requested
    if update_metadata and metadata_path.exists():
        update_metadata_csv(metadata_path, data_folder)
        print(f"\nUpdated {metadata_path}")
    elif update_metadata:
        print(f"\nWarning: metadata.csv not found at {metadata_path}")
    
    return converted_files, failed_files


def update_metadata_csv(metadata_path, data_folder):
    """
    Update metadata.csv to point to camera.json instead of fused_pose.npy.
    
    Args:
        metadata_path: Path to metadata.csv
        data_folder: Root data folder (for relative paths)
    """
    import csv
    import shutil
    
    # Create backup of original metadata
    backup_path = str(metadata_path) + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(metadata_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Read current metadata
    rows = []
    with open(metadata_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Update camera paths
    updated_count = 0
    for row in rows:
        if 'camera' in row and 'fused_pose.npy' in row['camera']:
            row['camera'] = row['camera'].replace('fused_pose.npy', 'camera.json')
            updated_count += 1
    
    # Write updated metadata
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {updated_count} camera paths in metadata.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Convert fused_pose.npy files to camera.json format'
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Root folder containing video_id/start_frame/ structure (default: ./data)'
    )
    parser.add_argument(
        '--single_file',
        type=str,
        default=None,
        help='Convert a single file (for testing)'
    )
    parser.add_argument(
        '--no_update_metadata',
        action='store_true',
        help='Skip updating metadata.csv'
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Convert single file
        output_path = convert_fused_pose_to_json(args.single_file)
        print(f"Converted {args.single_file} to {output_path}")
    else:
        # Process all files
        process_all_data(args.data_folder, update_metadata=not args.no_update_metadata)


if __name__ == '__main__':
    main()

