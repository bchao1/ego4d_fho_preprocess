import os
from argparse import ArgumentParser
import pandas as pd
import tqdm
import imageio
import numpy as np
from decord import VideoReader, cpu
import shutil
import multiprocessing
from functools import partial

def load_egovid5M_clips_annotations(egovid5M_folder):
    df_1 = pd.read_csv(os.path.join(egovid5M_folder, "egovid-kinematic.csv"))
    df_2 = pd.read_csv(os.path.join(egovid5M_folder, "egovid-val.csv"))
    df_3 = pd.read_csv(os.path.join(egovid5M_folder, "egovid-text.csv")) # Full 5M dataset, no poses
    df_clips = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df_clips = df_clips.rename(columns={"video_id": "clip_id"}) # original video_id is now clip_id = {video_id}_{start_frame}_{end_frame}
    df_clips["video_id"] = df_clips["clip_id"].str.rsplit("_", n=2).str[0] # video_id is the first part of the clip_id
    print("Found {} clips with frames annotations in the EgoVid-5M dataset\n".format(len(df_clips)))
    return df_clips

def load_ego4D_videos_with_poses(egovid5M_folder, ego4d_video_ids): 
    clips_with_poses = os.listdir(os.path.join(egovid5M_folder, "poses")) # {video_id}_{start_frame}_{end_frame}
    print("Found {} clips with poses in the EgoVid-5M dataset".format(len(clips_with_poses)))
    
    # Each clip might be extracted from the same video, so filter out unique video_ids
    video_ids = [video_id.split("_")[0] for video_id in clips_with_poses]
    video_ids = list(set(video_ids))
    print("Found {} unique video_ids in the EgoVid-5M dataset (with poses)".format(len(video_ids)))
    
    # Find the overlap between the two sets
    overlap_ids = list(set(video_ids) & set(ego4d_video_ids))
    print("Found {} overlap between the two sets\n".format(len(overlap_ids)))

    # Load one poses file and check pose shapes
    try:
        if len(clips_with_poses) > 0:
            test_extr = np.load(os.path.join(egovid5M_folder, "poses", clips_with_poses[0], "fused_pose.npy"))
            test_intr = np.load(os.path.join(egovid5M_folder, "poses", clips_with_poses[0], "intri.npy"))
            
            print(f"Test extrinsic shape: {test_extr.shape}")
            print(f"Test intrinsic shape: {test_intr.shape}")
    except Exception as e:
        print(f"Warning: Could not check pose shapes (might be empty?): {e}")
    
    return overlap_ids

def extract_clip_imageio(video_reader, start_frame, end_frame):
    extracted_clip = []
    for frame_id in range(start_frame, end_frame):
        extracted_clip.append(video_reader.get_data(frame_id))
    return np.array(extracted_clip)

def extract_clip_decord(video_reader, start_frame, end_frame):
    """
    Optimized decord version.
    video_reader: a decord.VideoReader object
    """
    # Create the list of indices you want to extract
    # decord's get_batch is significantly faster than a Python for-loop
    frame_indices = list(range(start_frame, end_frame))
    
    # Returns a decord NDArray, .asnumpy() converts it to a standard numpy array
    return video_reader.get_batch(frame_indices).asnumpy()

def process_clip_job(args_tuple):
    """
    Worker function to process a SINGLE CLIP.
    This opens the video, extracts one specific clip, and closes it.
    """
    video_id, clip_id, caption, input_folder, output_folder, egovid5M_folder, extraction_method = args_tuple

    # Note: Opening the video file for every single clip is less efficient for IO,
    # but allows perfect parallelization at the clip level as requested.
    video_path = os.path.join(input_folder, f"{video_id}.mp4")
    
    try:
        # robust parse: {video_id}_{start}_{end}
        base, start_s, end_s = clip_id.rsplit("_", 2)
        start_frame = int(start_s)
        end_frame = start_frame + 120 # only 120 frames for camera poses
        
        # Create output directory for this specific clip
        os.makedirs(os.path.join(output_folder, video_id, str(start_frame)), exist_ok=True)

        if extraction_method == "imageio":
            video_reader = imageio.get_reader(video_path, format="ffmpeg")
            fps = video_reader.get_meta_data()["fps"]
            extracted_clip = extract_clip_imageio(video_reader, start_frame, end_frame)
            video_reader.close()
        elif extraction_method == "decord":
            # Initialize VideoReader inside the process
            video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = video_reader.get_avg_fps()
            extracted_clip = extract_clip_decord(video_reader, start_frame, end_frame)
            del video_reader
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")

        # 1. Save the clip
        imageio.mimsave(
            os.path.join(output_folder, video_id, str(start_frame), 'video.mp4'),
            extracted_clip,
            fps=fps,
            macro_block_size=1
        )

        # 2. Save the caption
        with open(os.path.join(output_folder, video_id, str(start_frame), 'caption.txt'), 'w') as f:
            f.write(caption)
        
        # 3. Save the poses, only if they exist
        if os.path.exists(os.path.join(egovid5M_folder, "poses", clip_id.split(".")[0])):
            # Extrinsics
            shutil.copyfile(
                os.path.join(egovid5M_folder, "poses", clip_id.split(".")[0], "fused_pose.npy"),
                os.path.join(output_folder, video_id, str(start_frame), "fused_pose.npy")
            )
            # Intrinsics
            shutil.copyfile(
                os.path.join(egovid5M_folder, "poses", clip_id.split(".")[0], "intri.npy"),
                os.path.join(output_folder, video_id, str(start_frame), "intri.npy")
            )
            
    except Exception as e:
        print(f"Error processing clip {clip_id}: {e}")
        return clip_id # Return ID even on error to update progress
        
    return clip_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder (Ego4D subset you want to use for training)")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder (where you want to save the clips)")
    parser.add_argument("--egovid5M_folder", type=str, required=True, help="Path to the EgoVid-5M dataset metadata folder")
    parser.add_argument("--extraction_method", type=str, required=True, help="Method to extract the clips", choices=["imageio", "decord"])
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode, process only a few clips")
    parser.add_argument("--process_poses_clips_only", action="store_true", help="Process only the clips with poses")

    """
        - egovid-kinematic.csv: 68k videos, with accurate poses (use this)
        - egovid-text.csv: only text annoations (actual 5M), no poses
        - egovid-val.csv: validation set
    """

    args = parser.parse_args()

    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Ego4D 5M folder: {args.egovid5M_folder}")
    print(f"Parallel Workers: {args.num_workers}")
    os.makedirs(args.output_folder, exist_ok=True)

    print("Starting the preprocessing pipeline...\n")

    # 1. List all videos in the input folder (Ego4D subset you want to use)
    ego4d_video_ids = [f.split(".")[0] for f in os.listdir(args.input_folder) if f.endswith(".mp4")]
    print(f"Found {len(ego4d_video_ids)} videos in the input folder\n")

    # 2. Load EgoVid-5M poses annotations
    # (We still need this to filter which videos we care about)
    egovid5M_video_ids_with_poses = load_ego4D_videos_with_poses(args.egovid5M_folder, ego4d_video_ids)
    
    # Create a set for O(1) lookups during filtering
    if args.process_poses_clips_only:
        valid_video_ids = set(egovid5M_video_ids_with_poses) # only video ids with poses
    else:
        valid_video_ids = set(ego4d_video_ids) # all video ids
    
    print(f"Processing {len(valid_video_ids)} videos\n")

    # 3. Load EgoVid-5M clip frames
    egovid5M_clips_metadata = load_egovid5M_clips_annotations(args.egovid5M_folder)

    # 4. Extract clips (Fine-Grained Parallelization)
    
    print("Preparing tasks for parallel execution...")
    
    # Filter the metadata upfront to include ONLY the clips belonging to the videos we found
    valid_clips_df = egovid5M_clips_metadata[
        egovid5M_clips_metadata["video_id"].isin(valid_video_ids)
    ]
    
    print(f"Processing {len(valid_clips_df)} individual clips across {len(valid_video_ids)} videos.")

    tasks = []
    # Create a task for every single clip row
    for row in valid_clips_df.itertuples(index=False):
        clip_id = row.clip_id.strip()
        video_id = row.video_id.strip()
        caption = row.llava_cap.strip() if row.llava_cap else "" 
        
        tasks.append((
            video_id,
            clip_id,
            caption,
            args.input_folder,
            args.output_folder,
            args.egovid5M_folder,
            args.extraction_method
        ))
    
    if args.debug:
        print("Debug mode enabled: Processing only 5 clips.")
        tasks = tasks[:5]

    print(f"Dispatched {len(tasks)} clip tasks to {args.num_workers} workers.")

    # Use multiprocessing Pool
    # We parallelize over CLIPS now
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pbar = tqdm.tqdm(
            pool.imap_unordered(process_clip_job, tasks),
            total=len(tasks),
            desc="Clips Completed"
        )
        
        # Consume the iterator to trigger execution
        for _ in pbar:
            pass

    print("Preprocessing pipeline completed successfully\n")
