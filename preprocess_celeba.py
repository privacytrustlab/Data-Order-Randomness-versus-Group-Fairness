from torchvision import models, transforms
from PIL import Image
import os
import torch
from tqdm import tqdm
import numpy as np

model = models.resnet50(pretrained=True)
model_feat = torch.nn.Sequential(*list(model.children())[:-1])
model_feat.eval()

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' ## Select a subset of GPUs if needed
model_feat.cuda()

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )])

root_dir = 'data/celebA/img_align_celeba/'
os.makedirs(root_dir + 'feat_align_celeba/', exist_ok=True)

for filename in tqdm(os.listdir(root_dir + 'img_align_celeba/')):
    img = Image.open(root_dir + 'img_align_celeba/' + filename).convert('RGB')
    img_preprocessed = preprocess(img).cuda()
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    out = model_feat(batch_img_tensor).cpu().detach().numpy()

    with open(root_dir + 'feat_align_celeba/' + filename[:-3] + 'npy', 'wb') as f:
        np.save(f, out.squeeze())
