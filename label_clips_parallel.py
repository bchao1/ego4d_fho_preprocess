import os 
import sys

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class SuppressStderr:
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

import warnings
warnings.filterwarnings('ignore')

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import imageio

import json
import cv2
import csv
from argparse import ArgumentParser
import multiprocessing
import tqdm
from scipy.spatial.transform import Rotation

# --- Configuration ---
# Uses the standard/full model. If you downloaded the full model, change path here.
MODEL_PATH = 'hand_landmarker.task' 
# Print handled in main now to avoid spamming in parallel
# print(f"Using model from {MODEL_PATH}")

# Finger colors (RGB format)
FINGER_COLORS = {
    'pinky': [[100, 0, 100], [150, 0, 150], [200, 0, 200], [255, 0, 255]],    # magenta gradient
    'ring': [[0, 50, 100], [0, 75, 150], [0, 100, 200], [0, 125, 255]],       # blue gradient  
    'middle': [[0, 100, 50], [0, 150, 75], [0, 200, 100], [0, 255, 125]],     # green gradient
    'index': [[100, 100, 0], [150, 150, 0], [200, 200, 0], [255, 255, 0]],    # yellow gradient
    'thumb': [[100, 0, 0], [150, 0, 0], [200, 0, 0], [255, 0, 0]]                          # red gradient
}

# Landmark color (pure blue in RGB)
JOINT_RADIUS = 8
LINE_THICKNESS = 5
LANDMARK_COLOR = (100, 100, 100)  # Pure blue in RGB (BGR: (255, 0, 0))

CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
}

# --- Helper Functions ---

def landmarks_to_list(landmarks):
    return [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks]

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
        Path to created camera.json file, or None if conversion failed
    """
    try:
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
    except Exception as e:
        # Don't print in parallel mode to avoid spamming
        return None

def draw_depth_skeleton(height, width, hand_landmarks_list):
    """
    Draws a depth map with a continuous RGB colormap (Viridis):
    - Background is Black (0,0,0)
    - Close objects (low Z) are 'Hot'/Bright colors (e.g., Yellow in Viridis)
    - Far objects (high Z) are 'Cold'/Dark colors (e.g., Purple in Viridis)
    """
    depth_image = np.zeros((height, width, 3), dtype=np.uint8)

    if not hand_landmarks_list:
        return depth_image

    # 1. Generate the Colormap Lookup Table (LUT) once
    # We create a 1x256 image representing the gradient and apply the colormap
    # You can change cv2.COLORMAP_VIRIDIS to COLORMAP_JET, COLORMAP_PLASMA, etc.
    colormap_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_VIRIDIS)

    # 2. Find global Min/Max Z to normalize
    all_z = [lm.z for hand in hand_landmarks_list for lm in hand]
    if not all_z: 
        return depth_image
        
    min_z, max_z = min(all_z), max(all_z)
    range_z = max_z - min_z if max_z != min_z else 1.0

    def get_color(z_val):
        """Map Z value to BGR Color from LUT. Closer (Lower Z) = Higher Index (Brighter/Yellow)."""
        # Normalize 0 to 1
        norm = (z_val - min_z) / range_z
        
        # Invert so Close (Low Z) maps to index 255 (Yellow in Viridis)
        # and Far (High Z) maps to index 0 (Purple in Viridis)
        color_index = int((1 - norm) * 255)
        
        # Clamp index just in case
        color_index = max(0, min(255, color_index))
        
        # Grab color from LUT (returns a numpy array [B, G, R])
        b, g, r = colormap_lut[0, color_index]
        return (int(b), int(g), int(r))

    # 3. Draw Bones (Lines)
    for landmarks in hand_landmarks_list:
        px_points = [
            (int(lm.x * width), int(lm.y * height)) 
            for lm in landmarks
        ]

        for connection in solutions.hands.HAND_CONNECTIONS:
            idx1, idx2 = connection
            
            # Average depth for the bone to determine color
            z1 = landmarks[idx1].z
            z2 = landmarks[idx2].z
            avg_z = (z1 + z2) / 2.0
            
            color = get_color(avg_z)
            
            start_point = px_points[idx1]
            end_point = px_points[idx2]

            cv2.line(depth_image, start_point, end_point, color, 2)

    # 4. Draw Joints (Circles)
    for landmarks in hand_landmarks_list:
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * width), int(lm.y * height)
            color = get_color(lm.z)
            cv2.circle(depth_image, (cx, cy), 3, color, -1)

    return depth_image

def draw_skeleton_mask(height, width, hand_landmarks_list):
    """Draws colored skeletons on a black background with custom finger colors."""
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for landmarks in hand_landmarks_list:
        # Convert landmarks to pixel coordinates
        px_points = [
            (int(lm.x * width), int(lm.y * height)) 
            for lm in landmarks
        ]
        
        # Draw connections with finger-specific colors
        for finger, finger_connections in CONNECTIONS.items():
            for joint, (idx1, idx2) in enumerate(finger_connections):
                color_rgb = FINGER_COLORS[finger][joint]
                color = tuple(color_rgb)  # Use RGB directly
            
                start_point = px_points[idx1]
                end_point = px_points[idx2]
                cv2.line(mask_image, start_point, end_point, color, LINE_THICKNESS)
        
        # Draw landmarks as pure blue circles
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(mask_image, (cx, cy), JOINT_RADIUS, LANDMARK_COLOR, -1)
    
    return mask_image

def draw_rgb_annotated(rgb_image, hand_landmarks_list):
    """Draws hand landmarks with custom finger colors on the original image."""
    annotated_image = np.copy(rgb_image)
    h, w = rgb_image.shape[:2]
    
    for landmarks in hand_landmarks_list:
        # Convert landmarks to pixel coordinates
        px_points = [
            (int(lm.x * w), int(lm.y * h)) 
            for lm in landmarks
        ]
        
        # Draw connections with finger-specific colors
        for finger, finger_connections in CONNECTIONS.items():
            for joint, (idx1, idx2) in enumerate(finger_connections):
                color_rgb = FINGER_COLORS[finger][joint]
                color = tuple(color_rgb)  # Use RGB directly
            
                start_point = px_points[idx1]
                end_point = px_points[idx2]
                cv2.line(annotated_image, start_point, end_point, color, LINE_THICKNESS)
        
        # Draw landmarks as pure blue circles
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), JOINT_RADIUS, LANDMARK_COLOR, -1)
    
    return annotated_image

def process_video(video_path, output_root):
    # Construct Output Paths
    # Structure: output_root/video_id/start_frame/
    parts = video_path.split(os.sep)
    video_id = parts[-3]
    start_frame = parts[-2]
    
    # Create specific output dir
    save_dir = os.path.join(output_root, video_id, start_frame)
    os.makedirs(save_dir, exist_ok=True)
    
    path_rgb = os.path.join(save_dir, "annotated_rgb.mp4")
    path_mask = os.path.join(save_dir, "skeleton.mp4")
    path_depth = os.path.join(save_dir, "depth_map.mp4")
    path_json = os.path.join(save_dir, "landmarks.json")
    path_reference_img = os.path.join(save_dir, "reference_img.png")
    path_caption = os.path.join(save_dir, "caption.txt")
    path_fused_pose = os.path.join(save_dir, "fused_pose.npy")

    # In parallel, printing can get messy. 
    # We rely on the progress bar in main, but keeping this as debug info.
    # print(f"Processing: {video_id}/{start_frame}")

    # Initialize MediaPipe
    # Note: Must be initialized inside the worker process
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, 
        num_hands=3,  # Detect up to 3 hands
    )

    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta['fps']
        size = meta['size'] # (width, height)
        w, h = size
        
        writer_rgb = imageio.get_writer(path_rgb, fps=fps, codec='libx264', macro_block_size=1)
        writer_mask = imageio.get_writer(path_mask, fps=fps, codec='libx264', macro_block_size=1)
        writer_depth = imageio.get_writer(path_depth, fps=fps, codec='libx264', macro_block_size=1)
        
        results_dict = {}
        first_frame = None
        success = True
        
        with SuppressStderr():
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                for i, frame in enumerate(reader):
                    # Save first frame for reference image
                    if i == 0:
                        first_frame = frame
                    # MP requires RGB
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    timestamp_ms = int((i * 1000) / fps)
                    
                    # Detect
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    # success is false if more than 2 hands are detected
                    if len(detection_result.hand_landmarks) > 2:
                        success = False
                    
                    # Extract Landmarks (Standard Detection, No Force Swap)
                    current_hands_lms = detection_result.hand_landmarks
                    # 1. Create Annotated RGB
                    frame_rgb_out = draw_rgb_annotated(frame, current_hands_lms)
                    
                    # 2. Create Skeleton Mask (Black BG)
                    frame_mask_out = draw_skeleton_mask(h, w, current_hands_lms)
                    
                    # 3. Create Depth Map (Grayscale Interp)
                    frame_depth_out = draw_depth_skeleton(h, w, current_hands_lms)
                    
                    # Write Frames
                    writer_rgb.append_data(frame_rgb_out)
                    writer_mask.append_data(frame_mask_out)
                    writer_depth.append_data(frame_depth_out)
                    
                    # Store JSON Data
                    if current_hands_lms:
                        frame_data = []
                        for idx, hand_lms in enumerate(current_hands_lms):
                            # Get label if available
                            label = "Unknown"
                            if idx < len(detection_result.handedness):
                                label = detection_result.handedness[idx][0].category_name
                            
                            frame_data.append({
                                "label": label,
                                "landmarks": landmarks_to_list(hand_lms)
                            })
                        results_dict[i] = frame_data

        reader.close()
        writer_rgb.close()
        writer_mask.close()
        writer_depth.close()
        
        # Save first frame as reference image
        if first_frame is not None:
            # Convert RGB to BGR for cv2.imwrite (OpenCV uses BGR)
            first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path_reference_img, first_frame_bgr)
        
        with open(path_json, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Convert fused_pose.npy to camera.json if it exists
        path_camera_json = os.path.join(save_dir, "camera.json")
        if os.path.exists(path_fused_pose):
            convert_fused_pose_to_json(path_fused_pose, path_camera_json)
        
        # Return metadata dictionary
        return {
            "video_id": video_id,
            "start_frame": start_frame,
            "success": success,
            "error": None
        }
            
    except Exception as e:
        error_msg = str(e)
        return {
            "video_id": video_id,
            "start_frame": start_frame,
            "success": False,
            "error": error_msg
        }

def is_clip_processed(video_path, output_root):
    """
    Check if a clip has already been processed and all labeled files are valid.
    Returns True if all output files exist and are not corrupted:
    - annotated_rgb.mp4
    - skeleton.mp4
    - depth_map.mp4
    - landmarks.json
    - reference_img.png
    """
    try:
        # Construct output paths (same structure as process_video)
        parts = video_path.split(os.sep)
        video_id = parts[-3]
        start_frame = parts[-2]
        
        save_dir = os.path.join(output_root, video_id, start_frame)
        
        path_rgb = os.path.join(save_dir, "annotated_rgb.mp4")
        path_mask = os.path.join(save_dir, "skeleton.mp4")
        path_depth = os.path.join(save_dir, "depth_map.mp4")
        path_json = os.path.join(save_dir, "landmarks.json")
        path_reference_img = os.path.join(save_dir, "reference_img.png")
        
        # List of all required files
        required_files = [
            (path_rgb, "video"),
            (path_mask, "video"),
            (path_depth, "video"),
            (path_json, "json"),
            (path_reference_img, "image")
        ]
        
        # Check all files exist and are valid
        for file_path, file_type in required_files:
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            # Check if file size is reasonable (at least 1KB to avoid empty/corrupted files)
            if os.path.getsize(file_path) < 1024:
                return False
            
            # Validate file based on type
            if file_type == "video":
                # Try to verify the video is not corrupted by attempting to read it
                try:
                    reader = imageio.get_reader(file_path)
                    # Try to get metadata to verify it's readable
                    reader.get_meta_data()
                    reader.close()
                except Exception:
                    # If we can't read the video, consider it corrupted/not processed
                    return False
            elif file_type == "json":
                # Try to verify the JSON is valid by attempting to parse it
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                except Exception:
                    # If we can't parse the JSON, consider it corrupted/not processed
                    return False
            elif file_type == "image":
                # Try to verify the image is valid by attempting to read it
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        return False
                except Exception:
                    # If we can't read the image, consider it corrupted/not processed
                    return False
        
        # All files exist and are valid
        return True
    except Exception:
        # If any error occurs during checking, consider it not processed
        return False

def process_video_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing pool."""
    return process_video(*args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input data")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode, process only single video")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata CSV file (default: input_folder/metadata.csv)")
    parser.add_argument("--separate_failed", action="store_true", help="Separate successful and failed videos into different CSV files (default: False)")
    parser.add_argument("--no-skip-processed", dest="skip_processed", action="store_false", default=True, help="Process all clips even if already processed (default: skip processed)")
    args = parser.parse_args()

    input_folder = args.input_folder

    print(f"Using model from {MODEL_PATH}")
    print(f"Parallel Workers: {args.num_workers}")

    # Traverse directory structure as requested
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        exit()

    tasks = []
    skipped_videos = []  # Track skipped (already processed) videos for CSV

    print("Scanning directory for videos...")
    # Pre-scan directories to build task list
    for video_id in os.listdir(input_folder):
        video_id_path = os.path.join(input_folder, video_id)
        if not os.path.isdir(video_id_path): continue

        start_frames = os.listdir(video_id_path)
        start_frames.sort(key=lambda x: int(x) if x.isdigit() else x) # Robust sort
        
        for start_frame in start_frames:
            start_frame_path = os.path.join(video_id_path, start_frame)
            if not os.path.isdir(start_frame_path): continue
            
            video_path = os.path.join(start_frame_path, 'video.mp4')
            
            if not os.path.exists(video_path):
                print(f"Skipping {start_frame_path}, no video.mp4 found.")
                continue
            
            # Check if clip is already processed (all labeled files exist and are valid)
            if args.skip_processed and is_clip_processed(video_path, args.input_folder):
                skipped_videos.append({"video_id": video_id, "start_frame": start_frame})
                continue
            
            # We store the args tuple for each job
            tasks.append((video_path, args.input_folder))
    
    if args.skip_processed:
        print(f"Found {len(skipped_videos)} clips already processed. Processing {len(tasks)} remaining clips.")
    else:
        print(f"Processing {len(tasks)} clips (skipping disabled).")
    
    if args.debug:
        tasks = tasks[:1]
        args.num_workers = 1
        if tasks:
            print(tasks[0])

    print(f"Found {len(tasks)} videos to process.")

    # Execute jobs in parallel
    if tasks:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Use tqdm to show progress
            results = list(tqdm.tqdm(
                pool.imap_unordered(process_video_wrapper, tasks),
                total=len(tasks),
                desc="Labeling Videos"
            ))
    else:
        results = []
    
    # Set up CSV files
    if args.metadata_csv is None:
        metadata_csv_path = os.path.join(input_folder, "metadata.csv")
    else:
        metadata_csv_path = args.metadata_csv
    
    # Create CSV file(s) with headers
    fieldnames = ['video', 'reference_image', 'camera', 'control_video_new', 'prompt']
    csv_file = open(metadata_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    
    csv_file_failed = None
    csv_writer_failed = None
    metadata_failed_csv_path = None
    
    if args.separate_failed:
        # Create failed CSV path
        if args.metadata_csv is None:
            metadata_failed_csv_path = os.path.join(input_folder, "metadata_failed.csv")
        else:
            base_path, ext = os.path.splitext(metadata_csv_path)
            metadata_failed_csv_path = f"{base_path}_failed{ext}"
        
        csv_file_failed = open(metadata_failed_csv_path, 'w', newline='', encoding='utf-8')
        csv_writer_failed = csv.DictWriter(csv_file_failed, fieldnames=fieldnames)
        csv_writer_failed.writeheader()
    
    # Write CSV rows from collected results
    print("Writing metadata CSV files...")
    
    # Helper function to write a CSV row for a video
    def write_csv_row(video_id, start_frame, success, csv_writer, csv_writer_failed):
        # Check if required files exist
        save_dir = os.path.join(input_folder, video_id, start_frame)
        path_caption = os.path.join(save_dir, "caption.txt")
        path_camera_json = os.path.join(save_dir, "camera.json")
        
        # If camera.json doesn't exist, try to create it from fused_pose.npy
        if not os.path.exists(path_camera_json):
            path_fused_pose = os.path.join(save_dir, "fused_pose.npy")
            if os.path.exists(path_fused_pose):
                convert_fused_pose_to_json(path_fused_pose, path_camera_json)
        
        if not (os.path.exists(path_caption) and os.path.exists(path_camera_json)):
            return False
        
        # Create relative paths
        rel_video_path = os.path.join(".", video_id, start_frame, "video.mp4").replace(os.sep, "/")
        rel_reference_img = os.path.join(".", video_id, start_frame, "reference_img.png").replace(os.sep, "/")
        rel_camera = os.path.join(".", video_id, start_frame, "camera.json").replace(os.sep, "/")
        rel_control_video = os.path.join(".", video_id, start_frame, "skeleton.mp4").replace(os.sep, "/")
        rel_prompt = os.path.join(".", video_id, start_frame, "caption.txt").replace(os.sep, "/")
        
        row_data = {
            'video': rel_video_path,
            'reference_image': rel_reference_img,
            'camera': rel_camera,
            'control_video_new': rel_control_video,
            'prompt': rel_prompt
        }
        
        if csv_writer_failed is not None:
            # Separate CSV files mode
            if success:
                csv_writer.writerow(row_data)
            else:
                csv_writer_failed.writerow(row_data)
        else:
            # Single CSV file mode - write all rows
            csv_writer.writerow(row_data)
        return True
    
    # Write results from newly processed videos
    all_results = list(results) if results else []
    for result in tqdm.tqdm(all_results, desc="Writing CSV", leave=False):
        if result is None:
            continue
        
        video_id = result["video_id"]
        start_frame = result["start_frame"]
        success = result["success"]
        write_csv_row(video_id, start_frame, success, csv_writer, csv_writer_failed)
    
    # Write skipped (already processed) videos, assuming they're successful
    if skipped_videos:
        for skipped in tqdm.tqdm(skipped_videos, desc="Writing skipped videos to CSV", leave=False):
            video_id = skipped["video_id"]
            start_frame = skipped["start_frame"]
            write_csv_row(video_id, start_frame, True, csv_writer, csv_writer_failed)  # Assume successful
    
    csv_file.close()
    print(f"Metadata CSV saved to: {metadata_csv_path}")
    if csv_file_failed is not None:
        csv_file_failed.close()
        print(f"Failed metadata CSV saved to: {metadata_failed_csv_path}")
            
    print("Processing complete.")