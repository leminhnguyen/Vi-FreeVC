import torch
import os
from tqdm import tqdm

ssl_dir = "dataset/sr/wav2vec/hn_mp_vdts"
for ssl_file in tqdm(os.listdir(ssl_dir)):
    ssl_path = os.path.join(ssl_dir, ssl_file)
    ssl_content = torch.load(ssl_path)
    if ssl_content.shape[1] != 1024:
        print(ssl_path)
    # torch.save(ssl_content.transpose(1, 2), ssl_path)
