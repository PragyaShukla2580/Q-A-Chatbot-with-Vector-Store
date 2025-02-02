from fastapi import FastAPI
from route.api import router as query_router
from uvicorn import run
from route.store_api import router as fiass_router

from utils.logging import logger

logger.info("Starting FastAPI Application...")

app = FastAPI()
app.include_router(query_router)
app.include_router(fiass_router)

logger.info("FastAPI Service Started Successfully!")
if __name__ == "__main__":
    run(app,host="127.0.0.1", port= 8001)