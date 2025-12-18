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
    df_kinematic = pd.read_csv(os.path.join(egovid5M_folder, "egovid-kinematic.csv"))
    df_val = pd.read_csv(os.path.join(egovid5M_folder, "egovid-val.csv"))
    # df_text = pd.read_csv(os.path.join(egovid5M_folder, "egovid-text.csv")) # Full 5M dataset, no poses
    df_clips = pd.concat([df_kinematic, df_val], ignore_index=True)
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

def process_video_job(args_tuple):
    """
    Worker function to process a single video.
    Unpacks arguments and contains the logic previously inside the main loop.
    """
    video_id, input_folder, output_folder, egovid5M_folder, extraction_method, clips_in_video = args_tuple

    # Ensure output directory exists for this video
    os.makedirs(os.path.join(output_folder, video_id), exist_ok=True)
    
    video_path = os.path.join(input_folder, f"{video_id}.mp4")
    
    try:
        if extraction_method == "imageio":
            video_reader = imageio.get_reader(video_path, format="ffmpeg")
            fps = video_reader.get_meta_data()["fps"]
        elif extraction_method == "decord":
            # Initialize VideoReader inside the process to avoid pickle issues
            video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = video_reader.get_avg_fps()
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")

        # Note: In parallel execution, this inner tqdm might conflict visually with others.
        # We keep it as requested, but set position=1 to try and mitigate overlap, 
        # or rely on leave=False to clear it quickly.
        pbar_clips = tqdm.tqdm(
            clips_in_video.itertuples(index=False),
            total=len(clips_in_video),
            desc=f"Clips ({video_id})",
            leave=False,
            position=1, # Attempt to offset from main bar
            disable=False # Set to True if terminal output gets too messy
        )

        for row in pbar_clips:
            clip_id = row.clip_id.strip() # .mp4
            caption = row.llava_cap.strip() if row.llava_cap else "" # text caption for the clip. TODO: optional save out!

            # robust parse: {video_id}_{start}_{end}
            base, start_s, end_s = clip_id.rsplit("_", 2)
            start_frame = int(start_s)
            end_frame = start_frame + 120 # only 120 frames for camera poses
            os.makedirs(os.path.join(output_folder, video_id, str(start_frame)), exist_ok=True)

            pbar_clips.set_postfix_str(f"{clip_id}")

            if extraction_method == "imageio":
                extracted_clip = extract_clip_imageio(video_reader, start_frame, end_frame)
            elif extraction_method == "decord":
                extracted_clip = extract_clip_decord(video_reader, start_frame, end_frame)
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
            
            # 3. Save the poses
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
        
        # Cleanup
        if extraction_method == "imageio":
            video_reader.close()
        elif extraction_method == "decord":
            del video_reader
            
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return video_id # Return ID even on error to update progress
        
    return video_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder (Ego4D subset you want to use for training)")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder (where you want to save the clips)")
    parser.add_argument("--egovid5M_folder", type=str, required=True, help="Path to the EgoVid-5M dataset metadata folder")
    parser.add_argument("--extraction_method", type=str, required=True, help="Method to extract the clips", choices=["imageio", "decord"])
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers (defaults to CPU count)")

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
    egovid5M_video_ids_with_poses = load_ego4D_videos_with_poses(args.egovid5M_folder, ego4d_video_ids)

    # 3. Load EgoVid-5M clip frames
    egovid5M_clips_metadata = load_egovid5M_clips_annotations(args.egovid5M_folder)

    # 4. Extract clips (Parallelized)
    
    # Prepare arguments for each job. 
    # We pre-filter the dataframe so each worker only gets the rows relevant to its video.
    # This reduces overhead inside the worker.
    tasks = []
    print("Preparing tasks for parallel execution...")
    for video_id in egovid5M_video_ids_with_poses:
        # Filter metadata for this specific video
        clips_in_video = egovid5M_clips_metadata[
            egovid5M_clips_metadata["video_id"] == video_id
        ].copy() # copy to ensure it's picklable and independent
        
        if not clips_in_video.empty:
            tasks.append((
                video_id, 
                args.input_folder, 
                args.output_folder, 
                args.egovid5M_folder, 
                args.extraction_method, 
                clips_in_video
            ))

    print(f"Dispatched {len(tasks)} video tasks to {args.num_workers} workers.")

    # Use multiprocessing Pool
    # We use imap_unordered to update the progress bar as soon as ANY video finishes
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pbar_videos = tqdm.tqdm(
            pool.imap_unordered(process_video_job, tasks),
            total=len(tasks),
            desc="Videos Completed"
        )
        
        # Consume the iterator to trigger execution
        for _ in pbar_videos:
            pass

    print("Preprocessing pipeline completed successfully\n")