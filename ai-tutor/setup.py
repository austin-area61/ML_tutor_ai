from setuptools import setup, find_packages

setup(
    name='ai_tutor',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas==1.3.3',
        'scikit-learn==0.24.2',
        'numpy==1.21.2',
    ],
    entry_points={
        'console_scripts': [
            'data_preprocessing=ai_tutor.src.utils.data_preprocessing:main',
            'train_model=ai_tutor.models.training_script:main',
        ],
    },
    package_data={
        # If any package contains *.txt or *.csv files, include them:
        '': ['*.txt', '*.csv'],
    },
    include_package_data=True,
    description='AI Tutor Project for analyzing and predicting student performance',
    author='Austin Onyango',
    author_email='austinonyango.area61@gmail.com',
    url='https://github.com/austin-area61/ai_tutor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
