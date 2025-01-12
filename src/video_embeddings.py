import os
from tqdm import tqdm
import torch
from moviepy import VideoFileClip
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import yaml


def load_yaml(file_path: str) -> dict[dict[str, str]]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=200,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=DEVICE
)
YAML_PATH = 'params.yaml'
VARS = load_yaml(YAML_PATH)
WEIGHTS = ResNet18_Weights.DEFAULT
RESNET = resnet18(weights = WEIGHTS)
RESNET.eval()



def detect_face(frame):
    box, _ = DETECTOR.detect(frame)
    if box is not None:
        x, y, w, h = [int(coord) for coord in box[0]]
        face = frame[y:h, x:w]
        return face



def extract_frames(file_path):
    try:
        with VideoFileClip(file_path) as video:
            video = video.without_audio()

            frames = []
            for frame in video.iter_frames():
                face = detect_face(frame)
                
                if face is None:
                    continue

                frames.append(face)
                
            return frames
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")



class FeaturesExtractorCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(*list(RESNET.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        return torch.flatten(x, 1)



def get_video_embedding_cnn(file_path):
    try:
        file_emo_alias = file_path.split('/')[-1].split('_')[2]
        label = VARS['emotion_mapping'][file_emo_alias]
        
        frames = extract_frames(file_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        inputs = [transform(frame) for frame in frames]
        inputs = torch.stack(inputs).to(DEVICE)

        model = FeaturesExtractorCNN().to(DEVICE)
        model.eval()

        with torch.no_grad():
            embeddings = model(inputs).cpu()

        return embeddings.mean(dim=0), torch.tensor(label) 
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")



def get_all_video_embeddings(root = 'data/video_data'):
    embeddings = dict()
    paths = sorted([os.path.join(root, file) for file in os.listdir(root) if (file.endswith('.flv') and file != '1076_MTI_SAD_XX.flv')])
    for video_path in tqdm(paths):
        embedding, label = get_video_embedding_cnn(video_path)
        if embedding is not None:
            embeddings[embedding] = label
    return embeddings



def main() -> None:
    embeddings = get_all_video_embeddings('data/video_data')
    torch.save(embeddings, 'embeddings/video/raw2.pt')


if __name__ == '__main__':
    main()