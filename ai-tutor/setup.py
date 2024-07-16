$content = @"
from setuptools import setup, find_packages

setup(
    name='ai-tutor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'preprocess-data=src.utils.data_preprocessing:main',
            'train-model=models.training_script:main',
        ],
    },
)
"@
Set-Content -Path ai-tutor/setup.py -Value $content
