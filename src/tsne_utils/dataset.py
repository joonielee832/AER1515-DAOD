import torch
from PIL import Image
from torchvision import transforms

class TSNEDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, min_size, center_crop=False):
        super(TSNEDataset, self).__init__()
        self.img_list = img_list
        transformations = [transforms.ToTensor()]
        if center_crop:
            transformations.insert(0, transforms.CenterCrop(min_size))
        self.transformations = transforms.Compose(transformations)
        
        #! Resize vs center crop???

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file = self.img_list[idx]
        img = self.load_img(file)
        return self.transformations(img)
    
    @classmethod
    def load_img(cls, file):
        return Image.open(file).convert('RGB')