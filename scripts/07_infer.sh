source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

python convert.py \
    --hpfile logs/vi/config.json \
    --ptfile logs/vi/G_1940000.pth \
    --txtpath convert.txt \
    --outdir outputs/freevc