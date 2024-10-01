from setuptools import setup, find_packages

setup(
    name='peptideTransformer',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        # Add other dependencies here
        # TODO: Add the dependencies for the data_preprocess and data_analysis functions
    ],
    entry_points={
        'console_scripts': [
            'peptideTransformers=src.__main__:main',
        ],
    },
)