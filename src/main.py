import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.health import router as health_router
from src.api.search import router as search_router
from src.api.segmentation import router as segmentation_router
from src.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        description="QGraph AI backend bootstrap service",
    )

    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(segmentation_router)

    register_error_handlers(app)
    return app


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(
        _request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        detail: Any = exc.detail
        if not isinstance(detail, dict):
            detail = {"message": str(detail)}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "detail": detail},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        _request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": {"message": "Unexpected server error"},
            },
        )


app = create_app()
