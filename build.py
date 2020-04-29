#!/usr/bin/env python

import glob
import os
import tarfile

import urllib.request
from torch.utils.cpp_extension import CppExtension, include_paths


def download_extract(url, dl_path):
    if not os.path.isfile(dl_path):
        # Already downloaded
        urllib.request.urlretrieve(url, dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()


# Download/Extract openfst
download_extract(
    'http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.7.tar.gz', 'third_party/openfst-1.7.7.tar.gz'
)


# Does gcc compile with this header and library?
def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path \
              + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0


compile_args = ['-O3', '-std=c++1z', '-fPIC']
ext_libs = []
if compile_test('zlib.h', 'z'):
    compile_args.append('-DHAVE_ZLIB')
    ext_libs.append('z')

if compile_test('bzlib.h', 'bz2'):
    compile_args.append('-DHAVE_BZLIB')
    ext_libs.append('bz2')

if compile_test('lzma.h', 'lzma'):
    compile_args.append('-DHAVE_XZLIB')
    ext_libs.append('lzma')

third_party_libs = ["openfst-1.7.7/src/include", "ThreadPool"]
lib_sources = glob.glob('third_party/openfst-1.7.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode/src/*.cpp')

extension = CppExtension(
    name='ctcdecode._ext.ctc_decode',
    package=True,
    with_cuda=False,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes + include_paths(),
    libraries=ext_libs,
    extra_compile_args=compile_args,
    language='c++'
)
