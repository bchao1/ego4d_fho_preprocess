# Extract clips from videos
python extract_clips_parallel.py \
    --input_folder /scratch/m000051-pm04/brianchc/ego4d_fho/v2/full_scale \
    --output_folder /scratch/m000051-pm04/brianchc/handctrl_data_v1 \
    --egovid5M_folder /scratch/m000051-pm04/brianchc/egovid_5M/ \
    --extraction_method decord \
    --debug
