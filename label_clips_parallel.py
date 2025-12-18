import os 
import sys

os.environ['GLOG_minloglevel'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from argparse import ArgumentParser
import multiprocessing
import tqdm

# --- Configuration ---
# Uses the standard/full model. If you downloaded the full model, change path here.
MODEL_PATH = 'hand_landmarker.task' 
# Print handled in main now to avoid spamming in parallel
# print(f"Using model from {MODEL_PATH}")

# --- Helper Functions ---

def landmarks_to_list(landmarks):
    return [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks]

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
    """Draws white skeletons on a black background."""
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for landmarks in hand_landmarks_list:
        # Convert to protobuf for MP Utils
        proto_list = landmark_pb2.NormalizedLandmarkList()
        proto_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            image=mask_image,
            landmark_list=proto_list,
            connections=solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style()
        )
    return mask_image

def draw_rgb_annotated(rgb_image, hand_landmarks_list):
    """Draws standard MediaPipe annotations on the original image."""
    annotated_image = np.copy(rgb_image)
    
    for landmarks in hand_landmarks_list:
        proto_list = landmark_pb2.NormalizedLandmarkList()
        proto_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=proto_list,
            connections=solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style()
        )
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
    path_mask = os.path.join(save_dir, "skeleton_mask.mp4")
    path_depth = os.path.join(save_dir, "depth_map.mp4")
    path_json = os.path.join(save_dir, "landmarks.json")

    # In parallel, printing can get messy. 
    # We rely on the progress bar in main, but keeping this as debug info.
    # print(f"Processing: {video_id}/{start_frame}")

    # Initialize MediaPipe
    # Note: Must be initialized inside the worker process
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, 
        num_hands=2,
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

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            for i, frame in enumerate(reader):
                # MP requires RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestamp_ms = int((i * 1000) / fps)
                
                # Detect
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
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
        
        with open(path_json, 'w') as f:
            json.dump(results_dict, f, indent=4)
    
        return f"{video_id}/{start_frame}"
            
    except Exception as e:
        print(f"Error processing {video_id}/{start_frame}: {e}")
        return None

def process_video_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing pool."""
    return process_video(*args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input data")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    input_folder = args.input_folder

    print(f"Using model from {MODEL_PATH}")
    print(f"Parallel Workers: {args.num_workers}")

    # Traverse directory structure as requested
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        exit()

    tasks = []

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
            
            if os.path.exists(video_path):
                # We store the args tuple for each job
                tasks.append((video_path, args.input_folder))
            else:
                print(f"Skipping {start_frame_path}, no video.mp4 found.")

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
            
    print("Processing complete.")