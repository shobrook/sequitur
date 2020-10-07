import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 0, 0):
    print("Requires Python 3 to run.")
    sys.exit(1)

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="sequitur",
    description="Autoencoders for sequence data that work out-of-the-box",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v1.2.2",
    packages=["sequitur"],
    python_requires=">=3",
    url="https://github.com/shobrook/sequitur",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=['torch'],
    keywords=["sequitur", "autoencoder", "lstm", "sequence", "sequence-data", "seq2seq"],
    license="MIT"
)
