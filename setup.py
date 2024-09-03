from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

AUTHOR_NAME = 'PHILIP ODEY'
LIST_OF_REQUIREMENTS = ['streamlit']
SRC_REPO = 'src'

setup(
    name=SRC_REPO,  # The name of your package
    version="0.0.1",  # The initial release version
    description="A sample Python package for movies recommendation",  # Short description of your package
    author=AUTHOR_NAME,  # Your name as the package author
    author_email="philipodey75@gmail.com",  # Your email address
    packages=[SRC_REPO],  # Automatically find and include all packages
    install_requires=[LIST_OF_REQUIREMENTS],  # External dependencies that your package needs    ,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.12'
)
