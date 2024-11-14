from setuptools import setup, find_packages

setup(
    name='next_word_prediction_custom',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A custom next word prediction package',
    url='https://github.com/MaxwellFung/next_word_prediction_custom',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'torch>=1.7.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)