import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import gc
import os
from src.imagecolorization.logging import logger
from src.imagecolorization.entity.config_entity import DataTransformationConfig


class ImageColorizationDataset(Dataset):
    def __init__(self, dataset, image_size, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.image_size = tuple(image_size)  
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, idx):
        L = np.array(self.dataset[0][idx]).reshape(self.image_size)
        L = transforms.ToTensor()(L)
        
        ab = np.array(self.dataset[1][idx])
        ab = transforms.ToTensor()(ab)
        
        return ab, L
    
    
from torch.utils.data import DataLoader
import gc
import os
import numpy as np
import torch
from src.imagecolorization.logging import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def load_data(self):
        ab_df = np.load(self.config.data_path_black)[:self.config.DATA_RANGE]
        l_df = np.load(self.config.data_path_grey)[:self.config.DATA_RANGE]
        dataset = (l_df, ab_df)
        gc.collect()
        return dataset
    
    def get_datasets(self, dataset):
        train_dataset = ImageColorizationDataset(
            dataset=dataset,
            image_size=self.config.IMAGE_SIZE
        )
        test_dataset = ImageColorizationDataset(
            dataset=dataset,
            image_size=self.config.IMAGE_SIZE
        )
        
        return train_dataset, test_dataset
    
    def save_datasets(self, train_dataset, test_dataset):
        # Ensure the directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_dataset_path = os.path.join(self.config.root_dir, 'train_dataset.pt')
        test_dataset_path = os.path.join(self.config.root_dir, 'test_dataset.pt')

        try:
            # Save the datasets
            torch.save(train_dataset, train_dataset_path)
            torch.save(test_dataset, test_dataset_path)

            logger.info(f"Train dataset saved at: {train_dataset_path}")
            logger.info(f"Test dataset saved at: {test_dataset_path}")
        except Exception as e:
            logger.error(f"Error saving datasets: {str(e)}")
            raise e
    
    
        
        
    
    
    def save_dataloaders(self, train_loader, test_loader):
        # Ensure the directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_loader_path = os.path.join(self.config.root_dir, 'train_loader.pt')
        test_loader_path = os.path.join(self.config.root_dir, 'test_loader.pt')

        try:
            # Save the dataloaders
            torch.save(train_loader, train_loader_path)
            torch.save(test_loader, test_loader_path)

            logger.info(f"Train Loader saved at: {train_loader_path}")
            logger.info(f"Test Loader saved at: {test_loader_path}")
        except Exception as e:
            logger.error(f"Error saving dataloaders: {str(e)}")
            raise e
