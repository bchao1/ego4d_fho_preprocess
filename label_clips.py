import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import imageio
import os
import json
import cv2
import csv
from argparse import ArgumentParser
import tqdm

# --- Configuration ---
# Uses the standard/full model. If you downloaded the full model, change path here.
MODEL_PATH = 'hand_landmarker.task' 
print(f"Using model from {MODEL_PATH}")

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

def process_video(video_path, output_root, csv_writer, csv_writer_failed=None):
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

    # Initialize MediaPipe
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, 
        num_hands=3,  # Detect up to 10 hands (effectively all visible hands)
        # min_hand_detection_confidence=0.5,
        # min_hand_presence_confidence=0.5,
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
        
        # Write metadata row to CSV
        # Use relative paths from output_root
        rel_video_path = os.path.join(".", video_id, start_frame, "video.mp4").replace(os.sep, "/")
        rel_reference_img = os.path.join(".", video_id, start_frame, "reference_img.png").replace(os.sep, "/")
        rel_camera = os.path.join(".", video_id, start_frame, "fused_pose.npy").replace(os.sep, "/")
        rel_control_video = os.path.join(".", video_id, start_frame, "skeleton.mp4").replace(os.sep, "/")
        rel_prompt = os.path.join(".", video_id, start_frame, "caption.txt").replace(os.sep, "/")
        
        # Only write row if caption.txt and fused_pose.npy exist
        if os.path.exists(path_caption) and os.path.exists(path_fused_pose):
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
        else:
            print(f"Warning: Skipping CSV entry for {video_id}/{start_frame} - missing caption.txt or fused_pose.npy")
            
    except Exception as e:
        print(f"Error processing {video_id}/{start_frame}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input data")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata CSV file (default: input_folder/metadata.csv)")
    parser.add_argument("--separate_failed", action="store_true", help="Separate successful and failed videos into different CSV files (default: False)")
    args = parser.parse_args()

    input_folder = args.input_folder

    # Traverse directory structure as requested
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        exit()

    # Set up CSV files
    if args.metadata_csv is None:
        metadata_csv_path = os.path.join(input_folder, "metadata.csv")
    else:
        metadata_csv_path = args.metadata_csv
    
    # Create CSV file(s) with headers
    csv_file = open(metadata_csv_path, 'w', newline='', encoding='utf-8')
    fieldnames = ['video', 'reference_image', 'camera', 'control_video_new', 'prompt']
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

    # Pre-scan to collect all video paths for progress bar
    video_paths = []
    for video_id in os.listdir(input_folder):
        video_id_path = os.path.join(input_folder, video_id)
        if not os.path.isdir(video_id_path): continue

        start_frames = os.listdir(video_id_path)
        start_frames.sort(key=lambda x: int(x) if x.isdigit() else x)
        for start_frame in start_frames:
            start_frame_path = os.path.join(video_id_path, start_frame)
            if not os.path.isdir(start_frame_path): continue
            
            video_path = os.path.join(start_frame_path, 'video.mp4')
            if os.path.exists(video_path):
                video_paths.append((video_path, video_id, start_frame))

    # Process videos with progress bar
    try:
        pbar = tqdm.tqdm(video_paths, desc="Processing videos")
        for video_path, video_id, start_frame in pbar:
            pbar.set_description(f"Processing {video_id}/{start_frame}")
            process_video(video_path, args.input_folder, csv_writer, csv_writer_failed)
    finally:
        csv_file.close()
        print(f"Metadata CSV saved to: {metadata_csv_path}")
        if csv_file_failed is not None:
            csv_file_failed.close()
            print(f"Failed metadata CSV saved to: {metadata_failed_csv_path}")
