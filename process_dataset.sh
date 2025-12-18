# Extract clips from videos
python extract_clips.py \
    --input_folder /miele/data/ego4d_fho/v2/full_scale \
    --output_folder ./test_dataset/extracted_clips \
    --egovid5M_folder /miele/data/ego5M \
    --extraction_method decord # for faster extraction, use decord

# Extract poses from videos
python label_clips.py \
    --input_folder ./test_dataset/extracted_clips 