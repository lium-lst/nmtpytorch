# -*- coding: utf-8 -*-
import shutil
import pathlib
from setuptools import setup


def get_nmtpytorch_version():
    with open('nmtpytorch/__init__.py') as f:
        s = f.read().split('\n')[0]
        if '__version__' not in s:
            raise RuntimeError('Can not detect version from nmtpytorch/__init__.py')
        return eval(s.split(' ')[-1])


# Set up packages to install
packages = ['nmtpytorch']
packages.extend(['nmtpytorch.%s' % p for p in
                ('datasets', 'layers', 'metrics', 'models',
                 'samplers', 'utils')])
packages.extend(['nmtpytorch.cocoeval.%s' % m for m in
                 ('bleu', 'meteor', 'cider', 'rouge')])

if shutil.which('java') is None:
    print("*** WARNING: 'java' not found.")
    print("*** WARNING: You need to have JRE installed for METEOR to work.")


setup(
    name='nmtpytorch',
    version=get_nmtpytorch_version(),
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
    install_requires=[
        'numpy', 'scipy', 'scikit-learn', 'tqdm', 'pillow',
        'torch==0.3.1', 'torchvision==0.2.1',
        'sacrebleu==1.2.7', 'tensorboardX==1.1',
    ],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=packages,
    scripts=[str(p) for p in pathlib.Path('bin').glob('*')],
    zip_safe=False)
