from src.imagecolorization.conponents.model_trainer import ModelTrainer
from src.imagecolorization.config.configuration import ConfigurationManager


class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.load_datasets()
        model_trainer.create_dataloaders()
        model_trainer.initialize_model()
        model_trainer.train_model()
        model_trainer.save_model()
