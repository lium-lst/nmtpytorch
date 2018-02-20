# -*- coding: utf-8 -*-
import pathlib

from setuptools import setup
import nmtpytorch

# Set up packages to install
packages = ['nmtpytorch']
packages.extend(['nmtpytorch.%s' % p for p in
                ('datasets', 'layers', 'metrics', 'models',
                 'samplers', 'utils')])
packages.extend(['nmtpytorch.cocoeval.%s' % m for m in
                 ('bleu', 'meteor', 'cider', 'rouge')])


setup(
    name='nmtpytorch',
    version=nmtpytorch.__version__,
    description='Neural Machine Translation Framework in PyTorch',
    url='https://github.com/lium-lst/nmtpytorch',
    author='Ozan Caglayan, LIUM LST Team',
    author_email='ozancag@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
    ],
    keywords='nmt neural-mt translation deep-learning pytorch captioning',
    install_requires=['numpy', 'torch >= 0.3.0', 'tqdm', 'torchvision'],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=packages,
    scripts=[str(p) for p in pathlib.Path('bin').glob('*')],
    zip_safe=False)
