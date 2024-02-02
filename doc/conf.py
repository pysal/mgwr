# -*- coding: utf-8 -*-
#
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os
import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath('../'))

# import your package to obtain the version info to display on the docs website
try:
    # import your package
    import mgwr
except ImportError:
    # if your package is not installed, import it from the source directory
    # sys.path.insert(0, os.path.abspath('../mgwr'))
    import mgwr

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Enable google docstring style and numpy docstring style with napoleon
extensions = [  # 'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex', 'numpydoc',
    'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode',
    'sphinx.ext.mathjax', 'sphinx.ext.doctest', 'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx', 'matplotlib.sphinxext.plot_directive'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# for sphinxcontrib-bibtex
bibtex_bibfiles = ['_static/references.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = "mgwr"  # string of your project name, for example, 'giddy'
copyright = '2018, pysal developers'
author = 'pysal developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version.
version = mgwr.__version__  # should replace it with your PACKAGE_NAME
release = mgwr.__version__  # should replace it with your PACKAGE_NAME

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests/*']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_title = "%s v%s Manual" % (project, version)

# (Optional) Logo of your package. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
# html_logo = "_static/images/package_logo.jpg"

# (Optional) PySAL favicon
html_favicon = "_static/images/pysal_favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {

    # Navigation bar title. (Default: ``project`` value)
    'navbar_title':
    "mgwr",  # string of your project name, for example, 'giddy'

    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel':
    False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': True,
    # 'navbar_pagenav': False,

    # No sidebar
    'nosidebar':
    True,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 2,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden':
    "true",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar navbar-inverse",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top':
    "true",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position':
    'footer',

    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing (default) or the name of a valid theme
    # such as "amelia" or "cosmo", "yeti", "flatly".
    'bootswatch_theme':
    "yeti",

    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    'bootstrap_version':
    "5",

    # Navigation bar menu
    'navbar_links': [
        ("Installation", "installation"),
        ("API", "api"),
        ("References", "references"),
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}
# html_sidebars = {'sidebar': ['localtoc.html', 'sourcelink.html', 'searchbox.html']}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'mgwr' + 'doc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'mgwr.tex', u'mgwr Documentation', u'pysal developers',
     'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'mgwr', u'mgwr Documentation', [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'mgwr', u'mgwr Documentation', author, 'mgwr',
     'multiscale gwr', 'Miscellaneous'),
]

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True
class_members_toctree = True
numpydoc_show_inherited_class_members = True
numpydoc_use_plots = True

# display the source code for Plot directive
plot_include_source = True

#
# def setup(app):
#     app.add_stylesheet("pysal-styles.css")

# add style file to html output
html_css_files = ['pysal-styles.css']


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/3.10/': None}
