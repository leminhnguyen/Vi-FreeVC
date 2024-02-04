source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

# --min 68 --max 72 \
# --min 73 --max 76 \
# --min 77 --max 80 \
# --min 81 --max 84 \

python preprocess_sr.py \
    --min 85 --max 88 \
    --in_dir dataset/DUMMY_22k/vi \
    --wav_dir dataset/sr/wav \
    --ssl_dir dataset/sr/wavlm \
    --hf_version 2 \
    --wavlm_path wavlm/WavLM-Large.pt