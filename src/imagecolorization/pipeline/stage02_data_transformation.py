from src.imagecolorization.conponents.data_tranformation import DataTransformation
from src.imagecolorization.config.configuration import ConfigurationManager


class DataTransformationPipeline:
    def __init__(sefl):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
    
        # Load the dataset
        dataset = data_transformation.load_data()
        
        # Get the dataloader using the loaded dataset
        train_dataset, test_dataset = data_transformation.get_datasets(dataset)
    
        # Perform any further operations (e.g., saving the dataset)
        data_transformation.save_datasets(train_dataset, test_dataset)