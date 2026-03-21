import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "FiLM Explorer"
copyright = "2026, Groupe FiLM"
author = "Groupe FiLM"

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "fr"

html_theme = "alabaster"
html_static_path = ["_static"]
