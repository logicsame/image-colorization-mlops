from src.imagecolorization.conponents.model_evaluation import ModelEvaluation
from src.imagecolorization.config.configuration import ConfigurationManager


class ModelEvaluationPipeLine:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run()
        


    