from setuptools import setup, find_packages

# Check for valid Python version
if sys.version_info[:2] < (3, 0):
    print('MDMC requires Python 3.0 or better. Python {0:d}.{1:d}'
          ' detected'.format(*sys.version_info[:2]))

setup(
    name="PaiNN",
    version="1.0.0",
    description="Library for implementation of message passing neural networks in Pytorch",
    author="xinyang",
    author_email="xinyang@dtu.dk",
    url = "https://github.com/Yangxinsix/PaiNN-model",
    packages=["PaiNN"],
)
