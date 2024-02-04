import sys
sys.path.append('.')
import torch
from wavlm import WavLM, WavLMConfig

import torch
import librosa

def get_content(wavlm_model, wav_16khz):
    with torch.no_grad():
        c = wavlm_model.extract_features(wav_16khz.squeeze(1))[0]
    c = c.transpose(1, 2)
    return c


def get_wavlm_model():
    # load the pre-trained checkpoints
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    wavlm_model = get_wavlm_model().cuda()

    wav_rs = librosa.load("dataset/DUMMY_16k/vi/he_dp_news/he_dp_news_0000061.wav", sr=16000)[0]
    wav_16khz = torch.tensor(wav_rs).cuda().unsqueeze(0)
    content = get_content(wavlm_model.cuda(), wav_16khz)
    print(content.shape, content)

    # wav_dir = "dataset/sr/wav/hn_mp_vdts"
    # ssl_dir = "dataset/sr/wavlm/hn_mp_vdts"
    # os.makedirs(ssl_dir, exist_ok=True)

    # for wav_name in tqdm(os.listdir(wav_dir)):
    #     wav_path = os.path.join(wav_dir, wav_name)
    #     wav_rs = librosa.load(wav_path, sr=16000)[0]
    #     wav_16khz = torch.tensor(wav_rs).cuda().unsqueeze(0)
    #     content = get_content(wavlm_model.cuda(), wav_16khz)
    #     ssl_path = os.path.join(ssl_dir, wav_name.replace(".wav", ".pt"))
    #     torch.save(content, ssl_path)
    #     print(ssl_path)