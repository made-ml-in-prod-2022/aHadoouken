import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from utils import (
    load_model,
    obtain_df,
    InputDataModel,
    ModelResponse,
    ModelType,
)

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


model_lr = None
model_rf = None

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def startup():
    global model_lr, model_rf
    model_lr_path = os.getenv("PATH_TO_MODEL_LR")
    model_rf_path = os.getenv("PATH_TO_MODEL_RF")
    if model_lr_path is None:
        err = f"PATH_TO_MODEL_LR is None"
        logger.error(err)
        raise RuntimeError(err)
    if model_rf_path is None:
        err = f"PATH_TO_MODEL_RF is None"
        logger.error(err)
        raise RuntimeError(err)

    model_lr = load_model(model_lr_path)
    model_rf = load_model(model_rf_path)
    logger.info(msg="ML models were loaded")


@app.get("/healz")
def health(response_model=JSONResponse):
    if model_lr is None or model_rf is None:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({"status": "models are ready"}),
    )


@app.post("/predict", response_model=ModelResponse)
def predict(request: InputDataModel):
    data = obtain_df(request)
    logger.debug(msg=f"Received data:\n{data}")
    logger.info(msg=f"Predict using {request.model} model")
    if request.model == ModelType.lr:
        try:
            y_pred = model_lr.predict(data)
        except Exception:
            raise HTTPException(status_code=500, detail="Unable to predict")
    elif request.model == ModelType.rf:
        try:
            y_pred = model_rf.predict(data)
        except Exception:
            raise HTTPException(status_code=500, detail="Unable to predict")
    else:
        logger.error(msg=f"Specified invalid model {request.model}")
        raise HTTPException(status_code=400, detail="Invalid model type")
    logger.debug(msg=f"Calculated results: {y_pred}")
    return ModelResponse(predicted_values=[str(pred) for pred in y_pred])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
