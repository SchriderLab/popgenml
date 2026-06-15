# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from popgenml.data.stats import *

project = 'popgenml'
copyright = '2025, Dylan Ray'
author = 'Dylan Ray'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # For NumPy/Google style docstrings
    'sphinx.ext.autosummary',  # <-- Add this
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []
#autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']

autodoc_mock_imports = [
    "numpy",
    "skbio",
    "pkg_resources",
    "msprime",
    "tskit",
    "matplotlib",
    "pandas",
    "sklearn",       # scikit-learn
    "scipy",
    "networkx",
    "allel",         # scikit-allel
    "demesdraw",
    "seaborn",
    "tqdm",
    "seriate",
    "newick"
]
