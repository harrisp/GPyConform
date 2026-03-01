from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    readme = fh.read()

setup(
    name='gpyconform',
    version='0.2.0.post1',
    author='Harris Papadopoulos',
    author_email='h.papadopoulos@frederick.ac.cy',
    description='Extends GPyTorch with Gaussian Process Regression Conformal Prediction',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/harrisp/GPyConform',
    project_urls={
        'Bug Tracker': 'https://github.com/harrisp/GPyConform/issues',
        'Documentation': 'https://gpyconform.readthedocs.io',
    },
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        'gpytorch>=1.12,<1.15.2',
        'torch>=2.0',
        'linear-operator',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    keywords='Gaussian Process Regression, Conformal Prediction, Prediction Regions, Prediction Intervals, Uncertainty Quantification, Coverage Guarantee, Normalized Nonconformity'
)
