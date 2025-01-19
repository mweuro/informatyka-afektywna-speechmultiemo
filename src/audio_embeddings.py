import transformers
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, WavLMModel, HubertModel
from tqdm import tqdm
import torch
import librosa
import os
from extra import load_yaml, get_class_by_name



VARS = load_yaml('params.yaml')



def get_single_audio_embedding(file_path: str, 
                               model: transformers.models, 
                               processor: transformers.models) -> torch.tensor:
    
    waveform, sample_rate = librosa.load(file_path, sr = 16000)
    file_emo_alias = file_path.split('/')[-1].split('_')[2]
    label = VARS['emotion_mapping'][file_emo_alias]
    inputs = processor(waveform, sampling_rate = sample_rate, return_tensors = 'pt', padding = True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
    global_embedding = torch.mean(last_hidden_state, dim = 1)
    
    return global_embedding.squeeze(0), torch.tensor(label)



def get_all_audio_embeddings(root: str, 
                             model: transformers.models, 
                             processor: transformers.models) -> dict[torch.tensor, torch.tensor]:
    
    embeddings = dict()
    paths = sorted([os.path.join(root, file) for file in os.listdir(root) if (file.endswith('.wav') and file != '1076_MTI_SAD_XX.wav')])
    for path in tqdm(paths, desc="Processing audio files", unit="file"):
        embedding, label = get_single_audio_embedding(path, model, processor)
        embeddings[embedding] = label
    
    return embeddings



def main() -> None:
    
    audio_models = [VARS['audio_models'][0]]
    for d in audio_models:
        model = get_class_by_name('transformers', d['model']).from_pretrained(d['name'])
        processor = get_class_by_name('transformers', d['processor']).from_pretrained(d['name'])
        embeddings = get_all_audio_embeddings('data/audio_data', model, processor)
        torch.save(embeddings, f"embeddings/audio/raw_{d['alias']}.pt")



if __name__ == '__main__':
    main()