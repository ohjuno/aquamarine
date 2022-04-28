from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name                          = 'aquamarine',
    version                       = '0.0.1',
    author                        = 'jooh',
    author_email                  = 'jooh@vtouch.io',
    description                   = 'Object Detection Models - PyTorch',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    url                           = 'https://github.com/ohjuno/Aquamarine',
    packages                      = find_packages(),
    python_requires               = '>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
