import torch
from torch.utils.data import DataLoader
import mlflow
import dagshub
from tqdm.notebook import tqdm
import json
import os
import logging
from src.imagecolorization.conponents.model_building import Generator, Critic
from src.imagecolorization.conponents.model_trainer import CWGAN
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import pytorch_lightning as pl
from torchsummary import summary
from src.imagecolorization.conponents.model_building import Generator, Critic
from src.imagecolorization.conponents.data_tranformation import ImageColorizationDataset
from src.imagecolorization.logging import logger
import gc
import numpy as np
from src.imagecolorization.config.configuration import ConfigurationManager
from pathlib import Path


logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
import mlflow
import dagshub
from tqdm.notebook import tqdm
import json
import os
import logging
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
from src.imagecolorization.utils.common import save_json

logger = logging.getLogger(__name__)

class FID:
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception.eval()
        self.resize = transforms.Resize((299, 299))

    def convert_to_three_channels(self, images):
        if images.shape[1] == 2:
            images = torch.cat((images, images[:, :1, :, :]), dim=1)  # Duplicate one channel
        return images

    def preprocess_images(self, images):
        images = self.convert_to_three_channels(images)
        images = images.to(self.device)
        images = self.resize(images)
        return images

    def calculate_fid(self, real_images, generated_images):
        batch_size = 32
        real_features_list = []
        generated_features_list = []

        for i in range(0, len(real_images), batch_size):
            real_batch = self.preprocess_images(real_images[i:i+batch_size])
            generated_batch = self.preprocess_images(generated_images[i:i+batch_size])

            with torch.no_grad():
                real_features = self.inception(real_batch).view(real_batch.size(0), -1)
                generated_features = self.inception(generated_batch).view(generated_batch.size(0), -1)

            real_features_list.append(real_features.cpu())
            generated_features_list.append(generated_features.cpu())

        real_features = torch.cat(real_features_list, dim=0)
        generated_features = torch.cat(generated_features_list, dim=0)

        mu_diff = real_features.mean(dim=0) - generated_features.mean(dim=0)
        sigma_diff = real_features.std(dim=0) - generated_features.std(dim=0)

        fid = mu_diff.pow(2).sum() + sigma_diff.pow(2).sum()
        return fid.item()

class InceptionScore:
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception.eval()
        self.resize = transforms.Resize((299, 299))

    def convert_to_three_channels(self, images):
        if images.shape[1] == 2:  # If the input has 2 channels
            images = torch.cat((images, images[:, :1, :, :]), dim=1)  # Duplicate one channel
        return images

    def preprocess_images(self, images):
        images = self.convert_to_three_channels(images)
        images = images.to(self.device)
        images = self.resize(images)
        return images

    def calculate_is(self, images):
        batch_size = 1
        splits = 10
        preds = []

        for i in range(0, len(images), batch_size):
            batch = self.preprocess_images(images[i:i+batch_size])
            with torch.no_grad():
                pred = F.softmax(self.inception(batch), dim=1)
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        n_images = preds.shape[0]

        split_scores = []
        for k in range(splits):
            part = preds[k * (n_images // splits): (k + 1) * (n_images // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.critic = None

    def load_model(self):
        self.generator = Generator(input_channel=1, output_channel=2).to(self.device)
        self.critic = Critic(in_channels=3).to(self.device)

        self.generator.load_state_dict(torch.load(self.config.generator_model))
        self.critic.load_state_dict(torch.load(self.config.critic_model))

        self.generator.eval()
        self.critic.eval()

        logger.info("Model loaded successfully.")

    def load_data(self):
        self.test_dataset = torch.load(self.config.test_data)
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.all_params.BATCH_SIZE, 
            shuffle=True,
        )

    def evaluate_model(self):
        is_calculator = InceptionScore(self.device)
        fid_calculator = FID(self.device)

        all_preds = []
        all_real = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating", unit="batch"):
                real, condition = batch
                real, condition = real.to(self.device), condition.to(self.device)
                fake = self.generator(condition)
                all_preds.append(fake.cpu())
                all_real.append(real.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_real = torch.cat(all_real, dim=0)

        print("Calculating Inception Score for real images...")
        mean_real_is, std_real_is = is_calculator.calculate_is(all_real)
        print("Calculating Inception Score for generated images...")
        mean_fake_is, std_fake_is = is_calculator.calculate_is(all_preds)

        print("Calculating Fréchet Inception Distance...")
        fid_value = fid_calculator.calculate_fid(all_real, all_preds)

        results = {
            "inception_score_real": {"mean": float(mean_real_is), "std": float(std_real_is)},
            "inception_score_fake": {"mean": float(mean_fake_is), "std": float(std_fake_is)},
            "fid": float(fid_value)
        }
        return results

    def save_scores(self, results):
        save_json(path=Path('scores.json'), data=results)

    def log_to_mlflow(self, results):
        dagshub.init(repo_owner='HAKIM-ML', repo_name='image-colorization-mlops', mlflow=True)

        with mlflow.start_run():
            # Log all parameters
            for key, value in self.config.all_params.items():
                mlflow.log_param(key, value)

            # Log metrics
            mlflow.log_metric('inception_score_real_mean', results['inception_score_real']['mean'])
            mlflow.log_metric('inception_score_fake_mean', results['inception_score_fake']['mean'])
            mlflow.log_metric('fid', results['fid'])

            # Log the JSON file as an artifact
            mlflow.log_artifact('scores.json')

    def run(self):
        self.load_model()
        self.load_data()
        results = self.evaluate_model()
        self.save_scores(results)
        self.log_to_mlflow(results)
