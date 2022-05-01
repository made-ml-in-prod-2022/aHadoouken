import logging
import sys

logger = logging.getLogger(__name__)
# handler = logging.StreamHandler(sys.stdout)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
#print(logger.__dict__)
logger.info(msg=f"{1}asdasdsad")

def log():
    logger.info("An INFO message from " + __name__)
    #logger.error("An ERROR message from + " + __name__)