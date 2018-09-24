# -*- coding: utf-8 -*-
import shutil
import pathlib
import setuptools


def get_nmtpytorch_version():
    with open('nmtpytorch/__init__.py') as f:
        s = f.read().split('\n')[0]
        if '__version__' not in s:
            raise RuntimeError('Can not detect version from nmtpytorch/__init__.py')
        return eval(s.split(' ')[-1])


if shutil.which('java') is None:
    print("*** WARNING: 'java' not found.")
    print("*** WARNING: You need to have JRE installed for METEOR to work.")


with open('README.md') as f:
    long_description = f.read()


setuptools.setup(
    name='nmtpytorch',
    version=get_nmtpytorch_version(),
    description='Neural Machine Translation Framework in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lium-lst/nmtpytorch',
    author='Ozan Caglayan, LIUM LST Team',
    author_email='ozancag@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
    ],
    keywords='nmt neural-mt translation sequence-to-sequence deep-learning pytorch',
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'scipy', 'scikit-learn', 'tqdm', 'pillow',
        'torch', 'torchvision==0.2.1',
        'sacrebleu>=1.2.9', 'tensorboardX==1.1',
        'editdistance==0.4', 'ipdb', 'subword_nmt==0.3.5',
    ],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=setuptools.find_packages(),
    scripts=[str(p) for p in pathlib.Path('bin').glob('*')],
    zip_safe=False)
