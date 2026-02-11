from setuptools import setup, find_packages

setup(
    name='jarvis-ai-project',
    version='0.2.0',
    author='MastaTrill',
    author_email='contact@jarvisai.dev',
    description='A comprehensive AI/ML project focused on machine learning and deep learning development.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MastaTrill/JarvisAI',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=1.5.2',
        'pandas>=2.2.3',
        'numpy>=1.26.4',
        'opencv-python>=4.10.0',
        'mlflow>=2.18.0',
        'wandb>=0.18.7'
    ],
    extras_require={
        'deep-learning': [
            'torch>=2.5.0',
            'tensorflow>=2.18.0'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)