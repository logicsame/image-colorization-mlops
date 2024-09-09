from src.imagecolorization.constants import *
from src.imagecolorization.utils.common import read_yaml, create_directories
from src.imagecolorization.entity.config_entity import (DataIngestionConfig,
                                                        DataTransformationConfig,
                                                        ModelBuildingConfig,
                                                        ModelTrainerConfig,
                                                        ModelEvalutaionConfig)
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params
        
        data_transformation_cofig = DataTransformationConfig(
            root_dir= config.root_dir,
            data_path_black=config.data_path_black,
            data_path_grey=config.data_path_grey,
            BATCH_SIZE=params.BATCH_SIZE,
            IMAGE_SIZE=params.IMAGE_SIZE,
            DATA_RANGE=params.DATA_RANGE
        )
        
        return data_transformation_cofig
    
    def get_model_building_config(self) -> ModelBuildingConfig:
        config = self.config.model_building
        params = self.params

        model_building_config = ModelBuildingConfig(
            root_dir=Path(config.root_dir),
            KERNEL_SIZE_RES=params.KERNEL_SIZE_RES,
            PADDING=params.PADDING,
            STRIDE=params.STRIDE,
            BIAS=params.BIAS,
            SCALE_FACTOR=params.SCALE_FACTOR,
            DIM=params.DIM,
            DROPOUT_RATE=params.DROPOUT_RATE,
            KERNEL_SIZE_GENERATOR=params.KERNEL_SIZE_GENERATOR,
            INPUT_CHANNELS=params.INPUT_CHANNELS,
            OUTPUT_CHANNELS=params.OUTPUT_CHANNELS,
            IN_CHANNELS=params.IN_CHANNELS
        )
        return model_building_config
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        
        create_directories([config.root_dir])
        
        # Convert LEARNING_RATE to float explicitly
        learning_rate = float(params.LEARNING_RATE)
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            train_data_path=config.train_data_path,
            LEARNING_RATE=learning_rate,  # Use the converted float value
            LAMBDA_RECON=params.LAMBDA_RECON,
            DISPLAY_STEP=params.DISPLAY_STEP,
            IMAGE_SIZE=params.IMAGE_SIZE,
            INPUT_CHANNELS=params.INPUT_CHANNELS,
            OUTPUT_CHANNELS=params.OUTPUT_CHANNELS,
            EPOCH=params.EPOCH,
            BATCH_SIZE= params.BATCH_SIZE
        )
        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvalutaionConfig:
        config = self.config.model_evaluation 
        params = self.params

        model_evaluation_config = ModelEvalutaionConfig(
            
            test_data=config.test_data,
            generator_model=config.generator_model,
            critic_model=config.critic_model,
            all_params = params
            
        )

        return model_evaluation_config

        
    
    