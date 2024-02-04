from transformers import AutoProcessor, Wav2Vec2BertModel, AutoFeatureExtractor
import torch, librosa
import os, sys
from tqdm import tqdm

def get_wav2vec_model():
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(device)
    return processor, model

def get_content(processor, model, wav_16khz):
    inputs = processor(wav_16khz, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # print(outputs.extract_features, outputs.extract_features.shape)
    return last_hidden_states


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor, model = get_wav2vec_model()

    sampling_rate = 16000
    wav_rs = librosa.load("dataset/DUMMY_16k/vi/he_dp_news/he_dp_news_0000061.wav", sr=16000)[0]
    wav_16khz = torch.tensor(wav_rs).unsqueeze(0)
    content = get_content(processor, model, wav_16khz)
    print(content.shape, content)

    wav_dir = "dataset/sr/wav/hn_mp_vdts"
    ssl_dir = "dataset/sr/wav2vec/hn_mp_vdts"
    os.makedirs(ssl_dir, exist_ok=True)

    for wav_name in tqdm(os.listdir(wav_dir)):
        wav_path = os.path.join(wav_dir, wav_name)
        wav_rs = librosa.load(wav_path, sr=16000)[0]
        wav_16khz = torch.tensor(wav_rs).unsqueeze(0)
        content = get_content(processor, model, wav_16khz)
        ssl_path = os.path.join(ssl_dir, wav_name.replace(".wav", ".pt"))
        torch.save(content, ssl_path)
        print(ssl_path)


# from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
# import torch
# from datasets import load_dataset

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate

# processor = AutoProcessor.from_pretrained("facebook/w2v-bert-2.0")
# model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# # audio file is decoded on the fly
# inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)
