"""
Script to evaluate a model on a filelist.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from emonet.models import EmoNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nclasses",
    type=int,
    default=8,
    choices=[5, 8],
    help="Number of emotional classes to test the model on. Please use 5 or 8.",
)
parser.add_argument("--filelist", type=str, default="~/datasets/new_affectnet/test.txt", help="Path to the filelist.")
parser.add_argument(
    "--out_filelist", type=str, default="~/datasets/new_affectnet/test_out.txt", help="Path to the output filelist."
)
parser.add_argument("--use_gpu", action="store_true", help="Use GPU.")

args = parser.parse_args()

image_size = 256
n_expression = args.nclasses
transform_image = transforms.Compose([transforms.CenterCrop(image_size)])

device = "cuda:0" if args.use_gpu else "cpu"

state_dict_path = f"./pretrained/emonet_{n_expression}.pth"
print(f"Loading the model from {state_dict_path}.")
state_dict = torch.load(str(state_dict_path), map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

with open(args.filelist, "r") as f:
    lines = f.readlines()
videos = [line.strip() for line in lines]

emotion_dict_8 = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}
emotion_dict_5 = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear"}
emotion_dict = emotion_dict_8 if n_expression == 8 else emotion_dict_5
overal_mean_valence = 0.0
overal_mean_arousal = 0.0
emotion_list = []
with open(args.out_filelist, "w") as f:
    for video_path in videos:
        video = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
        video = video.to(device).float()
        video = transform_image(video) / 255
        with torch.no_grad():
            out = net(video)
        expr = out["expression"]
        valence = out["valence"]
        arousal = out["arousal"]
        most_present_emotion = emotion_dict[torch.bincount(torch.argmax(expr, dim=1)).argmax().item()]
        mean_valence = valence.mean().item()
        mean_arousal = arousal.mean().item()
        f.write(f"{video_path} {most_present_emotion} {mean_valence} {mean_arousal}\n")
        emotion_list.append(most_present_emotion)
        overal_mean_valence += mean_valence
        overal_mean_arousal += mean_arousal

    f.write(f"OVERALL Valence: {overal_mean_valence / len(videos)}\n")
    f.write(f"OVERALL Arousal: {overal_mean_arousal / len(videos)}\n")
    f.write(f"OVERALL Emotion: {max(set(emotion_list), key=emotion_list.count)}\n")
