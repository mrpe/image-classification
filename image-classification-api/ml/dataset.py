import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()

        self.resize = transforms.Compose([
            transforms.Resize((320, 320))
        ])

        with open(data_path, "r") as file:
            content = file.read()
            self.sample_list = content.split("\n")

    def __getitem__(self, index):

        try:
            image_path = self.sample_list[index]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 320))
            img = torch.from_numpy(img).float()
            img = np.transpose(img, (2, 0, 1))
        except Exception as e:
            print(e)
            print("Error:" + image_path)
            print("-------------------------------------")
            return None

        if ("Dog" in image_path):
            target = 1
        else:
            target = 0

        return img, target

    def __len__(self):
        return len(self.sample_list)


class SingleData(Dataset):

    def __init__(self, image_path):
        super().__init__()

        self.resize = transforms.Compose([
            transforms.Resize((320, 320))
        ])

        self.image_path = image_path

    def __getitem__(self, index):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = torch.from_numpy(img).float()
        img = np.transpose(img, (2, 0, 1))
        return img

    def __len__(self):
        return 1


class SingleByteData(Dataset):

    def __init__(self, image):
        super().__init__()

        self.resize = transforms.Compose([
            transforms.Resize((320, 320))
        ])

        self.image = image

    def __getitem__(self, index):
        file_bytes = np.asarray(bytearray(self.image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = torch.from_numpy(img).float()
        img = np.transpose(img, (2, 0, 1))
        return img

    def __len__(self):
        return 1
