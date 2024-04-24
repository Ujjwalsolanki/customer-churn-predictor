from logger import logging

STAGE_NAME = "Pipeline stage 1"

try:
    
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ## Create pipeline object and call it from here
    # object = ()
    # object.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
