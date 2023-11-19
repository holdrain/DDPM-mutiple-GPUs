# dataset classes
from torch.utils.data import Dataset
from torchvision import transforms as T
from functools import partial
from pathlib import Path
from utils.helper import exists,convert_image_to_fn
from torch import nn
from PIL import Image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 遍历一个或多个文件扩展名，查找一个文件夹（包括其所有子文件夹）中所有具有这些扩展名的文件，并将它们的路径作为一个列表返回
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # 如果convert_image_to非None则，用partial固定covert_image_to_fn的第一个参数，否则则设置其为恒等层
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            # T.Lambda用于定义用户自己的图像变换函数
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)