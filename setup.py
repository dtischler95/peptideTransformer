from setuptools import setup, find_packages

setup(
    name='peptideTransformer',
    version='0.1.0',
    description='Fine-tuned ProtBERT for hemolytic peptide classification and MIC regression',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.4.0',
        'transformers>=4.48.0',
        'accelerate>=0.34.0',
        'evaluate>=0.4.0',
        'datasets>=2.19.0',
        'scikit-learn>=1.5.0',
        'xgboost>=2.1.0',
        'umap-learn>=0.5.0',
        'numpy>=1.26.0',
        'pandas>=2.2.0',
        'scipy>=1.14.0',
        'matplotlib>=3.10.0',
        'seaborn>=0.13.0',
        'peptides>=0.3.0',
        'PyYAML>=6.0.0',
        'tqdm>=4.60.0',
    ],
    entry_points={
        'console_scripts': [
            'peptide-transformer=src.cli:main',
        ],
    },
)
