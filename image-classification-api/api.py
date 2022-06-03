
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utilities.utilities import get_uptime
from utilities.environment import Environment
from utilities.logging.config import initialize_logging, initialize_logging_middleware

from ml.emily import Emily
from static.render import render
from starlette.responses import HTMLResponse


emily = Emily()

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.

app = FastAPI()

initialize_logging()
initialize_logging_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainItem(BaseModel):
    dataset_path: str
    save_path: str


@app.post('/api/train')
def train(item: TrainItem):
    return {'result': emily.train(item)}


class EvaluateItem(BaseModel):
    dataset_path: str
    model_path: str


@app.post('/api/evaluate')
def evaluate(item: EvaluateItem):
    return {'result': emily.evaluate(item)}


class PredictItem(BaseModel):
    sample: str
    model_path: str


@app.post('/api/predict')
def predict(item: PredictItem):
    return {'result': emily.predict(item)}


@app.get('/api')
def hello():
    return {
        "service": Environment().COMPOSE_PROJECT_NAME,
        "uptime": get_uptime()
    }


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=Environment().HOST_IP,
            port=Environment().CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=Environment().HOST_IP,
        port=Environment().CONTAINER_PORT
    )

