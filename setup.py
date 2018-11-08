from setuptools import setup
from prwlr import __version__ as VERSION
from prwlr import __author__ as AUTHOR


setup(name="prwlr",
      version=VERSION,
      description="Module for annotatating genetic interactions networks with phylogenetic profiles.",
      author=AUTHOR,
      author_email="dariusz.izak@ibb.waw.pl",
      url="https://github.com/dizak/prwlr",
      license="MIT",
      py_modules=["prwlr"])
