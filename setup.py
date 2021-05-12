from setuptools import find_packages, setup
setup(name="pytorch_propane",
      version="0.1",
      description="High level framework for ML research on top of Pytorch. ",
      author="Divam Gupta",
      author_email='divamgupta@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="http://github.com/divamgupta/pytorch-propane",
      packages=find_packages(),
      )