# -*- coding: utf-8 -*-
import pathlib
import setuptools


def get_nmtpytorch_version():
    with open('nmtpytorch/__init__.py') as f:
        s = f.read().split('\n')[0]
        if '__version__' not in s:
            raise RuntimeError('Can not detect version from nmtpytorch/__init__.py')
        return eval(s.split(' ')[-1])


with open('README.md') as f:
    long_description = f.read()

with open('NEWS.md') as f:
    release_notes = f.read()

long_description = long_description.replace(
    '## Release Notes\n\nSee [NEWS.md](NEWS.md).\n', release_notes)

setuptools.setup(
    name='nmtpytorch',
    version=get_nmtpytorch_version(),
    description='Sequence-to-Sequence Framework in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lium-lst/nmtpytorch',
    author='Ozan Caglayan',
    author_email='ozancag@gmail.com',
    license='MIT',
    project_urls={
        'Wiki': 'https://github.com/lium-lst/nmtpytorch/wiki',
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX',
    ],
    keywords='nmt neural-mt translation sequence-to-sequence deep-learning pytorch',
    python_requires='~=3.7',
    install_requires=[
        'numpy', 'scikit-learn', 'tqdm', 'pillow',
        'torch==1.4.0', 'torchvision==0.5.0', 'pytorch-ignite==0.3.0',
        'sacrebleu>=1.2.9',
        'editdistance==0.4', 'subword_nmt==0.3.5',
    ],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=setuptools.find_packages(),
    scripts=[str(p) for p in pathlib.Path('bin').glob('*')],
    zip_safe=False)
