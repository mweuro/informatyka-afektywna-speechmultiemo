from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import os
import yaml


def load_yaml(file_path: str) -> dict[dict[str, str]]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


MODEL_NAME = 'facebook/wav2vec2-base'
PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
MODEL = Wav2Vec2Model.from_pretrained(MODEL_NAME)
YAML_PATH = 'params.yaml'
VARS = load_yaml(YAML_PATH)


def get_single_audio_embedding(file_path: str) -> torch.tensor:
    waveform, sample_rate = librosa.load(file_path, sr = 16000)
    file_emo_alias = file_path.split('/')[-1].split('_')[2]
    label = VARS['emotion_mapping'][file_emo_alias]
    inputs = PROCESSOR(waveform, sampling_rate = sample_rate, return_tensors = 'pt', padding = True)
    
    with torch.no_grad():
        outputs = MODEL(**inputs)
        last_hidden_state = outputs.last_hidden_state
    global_embedding = torch.mean(last_hidden_state, dim = 1)
    
    return global_embedding.squeeze(0), torch.tensor(label)


def get_all_audio_embeddings(root: str) -> dict[torch.tensor, torch.tensor]:
    embeddings = dict()
    paths = sorted([os.path.join(root, file) for file in os.listdir(root) if (file.endswith('.wav') and file != '1076_MTI_SAD_XX.wav')])
    
    for path in paths:
        embedding, label = get_single_audio_embedding(path)
        embeddings[embedding] = label
    
    return embeddings


def main() -> None:
    embeddings = get_all_audio_embeddings('data/audio_data')
    torch.save(embeddings, 'embeddings/audio/raw.pt')


if __name__ == '__main__':
    main()