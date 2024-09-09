from src.imagecolorization.conponents.model_building import ModelBuilding
from src.imagecolorization.config.configuration import ConfigurationManager

class ModelBuildingPipeline:
    def __init__(slef):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_config = config_manager.get_model_building_config()

        model_building = ModelBuilding(config=model_config)
        generator, critic = model_building.build_and_save()