import os
import random
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageFolderWithPath(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder to return (image, label, path).
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPath, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CovidDatasetLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def _get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _prepare_split(self):
        """
        Automatically splits flat COVID / non-COVID folders into train/val (80/20).
        """
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        # Check if already split
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            return train_dir, val_dir
            
        print("Train/Val split not found. Automatically splitting dataset 80/20...")
        
        classes = []
        if os.path.exists(self.data_dir):
            for d in os.listdir(self.data_dir):
                if d.lower() in ['covid', 'non-covid'] and os.path.isdir(os.path.join(self.data_dir, d)):
                    classes.append(d)
                
        if not classes:
            raise FileNotFoundError(f"Could not find COVID/non-COVID folders in {self.data_dir}. Are you passing the right directory?")
            
        for c in classes:
            src_dir = os.path.join(self.data_dir, c)
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Sort then shuffle with fixed seed to guarantee reproducibility
            files.sort()
            random.seed(42)
            random.shuffle(files)
            
            split_idx = int(0.8 * len(files))
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
            dest_c = 'Non-COVID' if c.lower() == 'non-covid' else 'COVID'
            train_dest = os.path.join(train_dir, dest_c)
            val_dest = os.path.join(val_dir, dest_c)
            
            os.makedirs(train_dest, exist_ok=True)
            os.makedirs(val_dest, exist_ok=True)
            
            for f in train_files:
                shutil.move(os.path.join(src_dir, f), os.path.join(train_dest, f))
            for f in val_files:
                shutil.move(os.path.join(src_dir, f), os.path.join(val_dest, f))
                
            # Remove empty source directory
            try:
                os.rmdir(src_dir)
            except OSError:
                pass
                
        print("Dataset successfully split into train/ and val/ folders!")
        return train_dir, val_dir

    def get_dataloaders(self):
        train_dir, val_dir = self._prepare_split()
        
        train_dataset = ImageFolderWithPath(root=train_dir, transform=self._get_train_transforms())
        val_dataset = ImageFolderWithPath(root=val_dir, transform=self._get_val_transforms())
        
        class_names = train_dataset.classes
        
        # num_workers=0 is required for reliable Windows operation
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        return train_loader, val_loader, class_names

def get_dataloaders(data_dir, batch_size=32):
    loader = CovidDatasetLoader(data_dir=data_dir, batch_size=batch_size)
    return loader.get_dataloaders()
