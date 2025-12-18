# Ego4D FHO dataset preprocessing scripts

## Environments 
```
conda create -n process_ego4D python=3.12 -y
conda activate process_ego4D 
python -m pip install -r requirements.txt
```

## Download Ego4D videos
1. Check Ego4D [website](https://ego4d-data.org/docs/start-here/) and setup AWS credentials and install the ego4d CLI.
2. Download the Ego4D dataset or a subset (here, I am using the FHO subset).
```bash
ego4d --output_directory="~/ego4d_data" --datasets full_scale # full dataset, full resolution
ego4d --output_directory="~/ego4d_data" --datasets full_scale --benchmarks FHO # FHO subset
```

## Download EgoVid-5M metadata and poses
1. Download data from [EgoVid-5M](https://github.com/JeffWang987/EgoVid).

## Run dataset preprocessing
```
./process_dataset.sh
```
This bash script does the following:

1. Extract EgoVid-5M clips from Ego4D videos 

```bash
python extract_clips.py \
    --input_folder /miele/data/ego4d_fho/v2/full_scale \ # path to the Ego4D dataset or subset
    --output_folder ./test_dataset/extracted_clips \ # path to the extracted clipsfolder
    --egovid5M_folder /miele/data/ego5M \ # path to the EgoVid-5M metadata folder
    --extraction_method decord # for faster extraction, use decord (default)
```

2. Label the clips with mediapipe
```bash
python label_clips.py \
    --input_folder ./test_dataset/extracted_clips # path to the extracted clips folder
```

## Processed dataset format
```
└── extracted_clips/
    ├── <video_id>/
        ├── <start_frame>/
            ├── video.mp4 # Raw video, [120, H, W, 3]
            ├── skeleton_mask.mp4 # (colored skeleton mask with black background)
            ├── annotated_rgb.mp4 # (colored skeleton overlayed with video)
            ├── depth_map.mp4 #(2D normalized depth map with black background)
            ├── caption.txt
            ├── fused_pose.npy # extrinsics, [120, 4, 4]
            ├── intri.npy # intrinsics [3, 3]
```