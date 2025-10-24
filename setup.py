from setuptools import setup, find_packages
from Cython.Build import cythonize
from pathlib import Path

py_files = [str(p) for p in Path("ot_dsim").rglob("*.py")]

setup(
    name='ot_dsim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pycryptodome',
        'tabulate',
    ],
    ext_modules=cythonize(
        py_files,
        compiler_directives={'language_level': "3"},
    ),
    zip_safe=False,
)