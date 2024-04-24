from logger import logging
from src.components.data_preprocessing import DataPreprocessing

STAGE_NAME = "Pipeline stage 1"

try:
    
    logging.info("\nx==========x\n")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    # Create pipeline object and call it from here
    object = DataPreprocessing()
    object.initiate_data_preprocessing()

    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
