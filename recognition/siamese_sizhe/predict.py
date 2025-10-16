# 载入最优权重，对一对图片输出相似度概率
import os, argparse
import torch
from PIL import Image
import torchvision.transforms as T
from modules import SiameseNet
import torch.nn.functional as F

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--img1", type=str, required=True)
    ap.add_argument("--img2", type=str, required=True)
    ap.add_argument("--size", type=int, default=224)
    return ap.parse_args()

def load_image(path, size):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    x1 = load_image(args.img1, args.size).to(device)
    x2 = load_image(args.img2, args.size).to(device)
    with torch.no_grad():
        logits = model(x1, x2)
        prob = torch.sigmoid(logits).item()
    print(f"Similarity probability: {prob:.4f}")

if __name__ == "__main__":
    main()
