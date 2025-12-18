# Extract poses from videos
mode=$1 # debug (lab servers) or full (marlowe)

if [ "$mode" == "full" ]; then
python label_clips_parallel.py \
    --input_folder /scratch/m000051-pm04/brianchc/handctrl_data_v1
else
python label_clips_parallel.py \
    --input_folder ./test_dataset/handctrl_data_v1
fi

