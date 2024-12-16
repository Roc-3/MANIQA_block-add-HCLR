import os
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as F

ROOT = Path(os.getcwd()) / 'test'
TARGET = Path('../../UIEB_resize') / 'test'
os.makedirs(TARGET, exist_ok=True)
folders = os.listdir(ROOT)
ims = [_ for _ in folders if _.endswith(('.png', '.jpg'))]
for im in ims:
    img = Image.open(ROOT / im)
    img = F.resize(img, (256, 256))
    img.save(TARGET / im)
    print(TARGET / im)
