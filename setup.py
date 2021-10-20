from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='physion',
    version='1.0',
    description='Vision Physiology - Code for experimental setup and analysis to study the physiology of visual circuits',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yzerlaut/physion',
    author='Yann Zerlaut',
    author_email='yann.zerlaut@icm-institute.org',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='vision physiology',
    packages=find_packages(),
    install_requires=[
        "pynwb",
        "scipy",
        "numpy",
        "argparse",
        "pyqt5",
        "pyqtgraph",
        "pathlib"
    ]
)
