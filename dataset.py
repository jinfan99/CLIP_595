import torch
from torch import nn
from torch.utils.data import Dataset
import os.path as osp
from glob import glob
from PIL import Image
import random

IMAGE_ROOT_DIR = "/home/zjf/repos/proj_595/data/action_effect_image_rs"

class SimpleDataset(Dataset):
    def __init__(self, root_dir=IMAGE_ROOT_DIR, transform=None, id_list=None):
        self.root_dir = root_dir
        self.transform = transform
        self.id_list = id_list
        assert id_list is not None
        self.data_list = []
        self.class2idx = {}
        self.idx2class = {}
        
        effect_dir_list = glob(osp.join(root_dir, '*'))
        class_counter = 0
        for effect_dir in effect_dir_list:
            effect_name = osp.basename(effect_dir)
            self.class2idx[effect_name] = class_counter
            self.idx2class[class_counter] = effect_name
            class_counter += 1
            
            img_path_list = glob(osp.join(effect_dir, 'positive', '*.jpeg'))
            self.data_list += img_path_list
            
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        file_name = self.data_list[self.id_list[idx]]
        image = Image.open(file_name).convert("RGB")
        image_class = file_name.split('/')[-3]
        class_idx = self.class2idx[image_class]
        
        item_dict = {}
        item_dict['pixel_values'] = self.transform(image)
        item_dict['label'] = class_idx
        # item_dict['image'] = image
        
        return item_dict
    
class CLIPDataset(Dataset):
    def __init__(self, processor, root_dir=IMAGE_ROOT_DIR, max_target_length=5, id_list=None):
        self.root_dir = root_dir
        self.data_list = []
        self.processor = processor
        self.max_target_length = max_target_length
        
        self.id_list = id_list
        
        self.class2idx = {}
        self.idx2class = {}
        
        effect_dir_list = glob(osp.join(root_dir, '*'))
        class_counter = 0
        for effect_dir in effect_dir_list:
            effect_name = osp.basename(effect_dir)
            self.class2idx[effect_name] = class_counter
            self.idx2class[class_counter] = effect_name
            class_counter += 1
            
            img_path_list = glob(osp.join(effect_dir, 'positive', '*.jpeg'))
            self.data_list += img_path_list
            
        # self.data_list = self.data_list[:10]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        file_name = self.data_list[self.id_list[idx]]
        image = Image.open(file_name).convert("RGB")
        
        image_class = file_name.split('/')[-3]
        text = image_class.replace('+', ' ')
        # print(text)

        pixel_values = self.processor.feature_extractor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, 
                                padding="max_length", 
                                max_length=self.max_target_length,
                                truncation=True).input_ids
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # print(labels)
        return {"input_ids":torch.tensor(labels), "pixel_values":pixel_values.squeeze()}
    
def make_dataset(transform, root_dir=IMAGE_ROOT_DIR, val_ratio=0.2):
    N = 1547
    num_val = int(N * val_ratio)
    val_idx = random.sample(range(N), num_val)
    train_idx = [idx for idx in range(N) if idx not in val_idx]
    
    train_ds = SimpleDataset(transform=transform, id_list=train_idx)
    val_ds = SimpleDataset(transform=transform, id_list=val_idx)
    
    return train_ds, val_ds

def make_dataset_clip(processor, root_dir=IMAGE_ROOT_DIR, val_ratio=0.2):
    N = 1547
    num_val = int(N * val_ratio)
    val_idx = random.sample(range(N), num_val)
    train_idx = [idx for idx in range(N) if idx not in val_idx]
    
    train_ds = CLIPDataset(processor, id_list=train_idx)
    val_ds = CLIPDataset(processor, id_list=val_idx)
    
    return train_ds, val_ds