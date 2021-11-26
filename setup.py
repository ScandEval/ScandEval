from setuptools import setup, find_packages
from pathlib import Path
from bump_version import get_current_version

PYTORCH_REQUIRES = ['torch>=1.6.0']
SPACY_REQUIRES = ['spacy>=3.2.0', 'spacy-transformers>=1.1.0']
ALL_REQUIRES = (PYTORCH_REQUIRES +
                SPACY_REQUIRES)

setup(name='scandeval',
      version=get_current_version(return_tuple=False),
      description='',
      long_description=Path('README.md').read_text(),
      long_description_content_type='text/markdown',
      url='https://github.com/saattrupdan/scandeval',
      author='Dan Saattrup Nielsen',
      author_email='saattrupdan@gmail.com',
      license='MIT',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'],
      packages=find_packages(exclude=('tests',)),
      include_package_data=True,
      install_requires=['numpy>=1.19.2',
                        'transformers>=4.5.0',
                        'datasets>=1.11.0',
                        'requests>=2.19.0',
                        'tqdm>=4.62.3',
                        'sentencepiece>=0.1.96',
                        'seqeval>=1.0.0',
                        'bs4>=0.0.1',
                        'termcolor>=1.0.0'],
      extras_require=dict(pytorch=PYTORCH_REQUIRES,
                          spacy=SPACY_REQUIRES,
                          all=ALL_REQUIRES),
      entry_points=dict(console_scripts=['scandeval=scandeval.cli:benchmark']))
