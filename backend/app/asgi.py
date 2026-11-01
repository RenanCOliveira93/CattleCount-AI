"""Ponto de entrada ASGI: uvicorn backend.app.asgi:app"""
from .main import create_app

app = create_app()
