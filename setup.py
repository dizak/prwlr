from setuptools import setup
from prowler import __version__ as VERSION
from prowler import __author__ as AUTHOR


setup(name="prowler",
      version=VERSION,
      description="Module for annotatating genetic interactions networks with phylogenetic profiles.",
      author=AUTHOR,
      author_email="dariusz.izak@ibb.waw.pl",
      url="https://github.com/dizak/prowler",
      license="MIT",
      py_modules=["prowler"])
