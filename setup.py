"""Setup.py for OpenEA."""

import os

import setuptools

MODULE = 'muKG'
VERSION = '1.0'
PACKAGES = setuptools.find_packages(where='src')
META_PATH = os.path.join('src', MODULE, '__init__.py')
KEYWORDS = ['Knowledge Graph', 'Embeddings']
INSTALL_REQUIRES = [
    # 'tensorflow',
    'tqdm',
    'joblib',
    'pandas',
    'matching==0.1.1',
    'scipy',
    'scikit-learn',
    'numpy',
    'gensim',
    'python-Levenshtein',
    'scipy',
    'redis',
]

if __name__ == '__main__':
    setuptools.setup(
        name=MODULE,
        version=VERSION,
        description='An open-source Python library for representation learning over knowledge graphs.',
        url='https://github.com/nju-websoft/muKG.git',
        author='Zequn Sun',
        author_email='zqsun.nju@gmail.com',
        maintainer='Zequn Sun',
        maintainer_email='zqsun.nju@gmail.com',
        keywords=KEYWORDS,
        packages=setuptools.find_packages(where='src'),
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        zip_safe=False,
    )
