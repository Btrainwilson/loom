# Configuration file for the Sphinx documentation builder.

project = "Loom"
copyright = "2025, Blake A. Wilson"
author = "Blake A. Wilson"

extensions = ["myst_parser", "autodoc2"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc2_render_plugin = "myst"

autodoc2_packages = [
    {
        "path": "../loomlib",
        "exclude_dirs": ["templates"],
        "auto_mode": False,
    }
]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
