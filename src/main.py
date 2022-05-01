import click
import logging
import sys

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option("--mode", required=True, type=str, help="train/predict")
@click.option("--config", type=str, help="path to config (for train mode)")
@click.option("--model", type=str, help="path to model (for predict mode)")
@click.option("--data", type=str, help="path to data (for predict mode)")
def main(mode, config, model, data):
    logger.info(msg="aaaa")
    models.train_model.log()


if __name__=="__main__":
    main()