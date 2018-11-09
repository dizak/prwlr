from distutils.core import setup
from setuptools import find_packages
import subprocess as sp


VERSION = sp.check_output('git tag -l', shell=True).split()[-1]

setup(
    name="prwlr",
    version=VERSION,
    author='Dariusz Izak IBB PAS',
    packages=find_packages(exclude=["*test*"]),
    install_requires=open("requirements.txt").readlines(),
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
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)
