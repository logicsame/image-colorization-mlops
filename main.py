from src.imagecolorization.pipeline.stage01_data_ingestion import DataIngestionPipeline
from src.imagecolorization.pipeline.stage02_data_transformation import DataTransformationPipeline
from src.imagecolorization.pipeline.stage_03_model_building import ModelBuildingPipeline
from src.imagecolorization.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from src.imagecolorization.pipeline.stage_05_model_evaluation import ModelEvaluationPipeLine
from src.imagecolorization.logging import logger

STAGE_NAME = 'Data Ingestion Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
STAGE_NAME = 'Data Tranasformation Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
STAGE_NAME = 'Model Building Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_building = ModelBuildingPipeline()
   model_building.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
STAGE_NAME = 'Model Training Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_trianer = ModelTrainerPipeline()
   model_trianer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



# STAGE_NAME = 'Model Evaluation Stage'

# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trianer = ModelEvaluationPipeLine()
#    model_trianer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

