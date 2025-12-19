# Extract clips from videos
mode=$1 # debug (lab servers) or full (marlowe)
out_folder=$2

if [ "$mode" == "full" ]; then
python extract_clips_parallel.py \
    --input_folder /scratch/m000051-pm04/brianchc/ego4d_fho/v2/full_scale \
    --output_folder /scratch/m000051-pm04/brianchc/$out_folder \
    --egovid5M_folder /scratch/m000051-pm04/brianchc/egovid_5M/ \
    --extraction_method decord \
    --num_workers 16
else
python extract_clips_parallel.py \
    --input_folder /miele/data/ego4d_fho/v2/full_scale/ \
    --output_folder ./test_dataset/handctrl_data_v1 \
    --egovid5M_folder /miele/data/ego5M/ \
    --extraction_method decord
fi

