from setuptools import setup, find_packages

setup(
    name="movie_recommender",
    version="0.1.0",
    author="Philip Odey",
    author_email="philipodey75@gmail.com",
    description="A Streamlit-based movie recommender system using TMDb API",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Philipodey/movie_recommender.git",
    packages=find_packages(),
    install_requires=[
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.2.2',
        'streamlit==1.30.0',
        'requests==2.31.0',
    ],
    entry_points={
        'console_scripts': [
            r'movie_recommender\recommender_system=movie_recommender\recommender_system:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
