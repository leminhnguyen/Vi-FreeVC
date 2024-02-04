source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

python train.py \
    --config configs/freevc.json \
    --model vi-mp-wav2vec