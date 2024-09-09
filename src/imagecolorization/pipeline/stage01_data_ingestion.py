from src.imagecolorization.conponents.data_ingestion import DataIngestion
from src.imagecolorization.config.configuration import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingesion = DataIngestion(config=data_ingestion_config)
        data_ingesion.load_file()
        data_ingesion.extract_zip_file()