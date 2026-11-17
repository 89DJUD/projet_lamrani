"""Point d’entrée de l’assistant pour les migrants."""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_module("migrant_ai.interface", run_name="__main__")
