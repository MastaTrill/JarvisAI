from setuptools import setup, find_packages

setup(
    name='jarvis-ai-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive AI/ML project focused on machine learning and deep learning development.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/jarvis-ai-project',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow',
        'torch',
        'scikit-learn',
        'pandas',
        'numpy',
        'opencv-python',
        'mlflow',
        'wandb'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)