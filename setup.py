from distutils.core import setup
from setuptools import find_packages
from prwlr import __version__
from prwlr import __author__


VERSION = __version__
AUTHOR = __author__

setup(
    name="prwlr",
    version=VERSION,
    author=AUTHOR,
    packages=find_packages(exclude=["*test*"]),
    install_requires=open("requirements.txt").readlines(),
    python_requires='>3.5',
    long_description=open("README.md").read(),
    description="Annotate genetic interactions networks with phylogenetic profiles and other attributes.",
    author_email="dariusz.izak@ibb.waw.pl",
    url="https://github.com/dizak/prwlr",
    license="BSD",
    keywords=[
        'genetic',
        'interactions',
        'network',
        'phylogenetic',
        'profile',
        'module',
        'pathway',
        'kegg',
        'evolution',
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)
