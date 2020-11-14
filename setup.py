#!/usr/bin/env python
import glob
import os
import tarfile
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile, naive_recompile
from setuptools.command.build_ext import build_ext

__version__ = '0.4.4'

# first, download and extract openfst


def download_extract(url, dl_path):
    from urllib.request import urlretrieve

    if not os.path.isfile(dl_path):
        # Already downloaded
        urlretrieve(url, dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()


if not os.path.isdir('third_party'):
    os.mkdir('third_party')
if not os.path.isdir('ctcdecode/_ext'):
    os.mkdir('ctcdecode/_ext')
    with open('ctcdecode/_ext/__init__.py', 'w'):
        pass

download_extract(
    'http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.7.tar.gz', 'third_party/openfst-1.7.7.tar.gz'
)

third_party_libs = ["openfst-1.7.7/src/include"]
lib_sources = glob.glob('third_party/openfst-1.7.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode/src/*.cpp')

ext_modules = [
    Pybind11Extension(
        "ctcdecode._ext.ctc_decode",
        sources=ctc_sources + lib_sources,
        include_dirs=third_party_includes,
        language='c++',
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

setup(
    name="ctcdecode",
    version=__version__,
    description="CTC Decoder for PyTorch based on Paddle Paddle's implementation",
    url="https://github.com/parlance/ctcdecode",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
