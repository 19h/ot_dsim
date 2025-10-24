from setuptools import setup, find_packages

setup(
    name='ot_dsim',
    version='0.1.0',
    packages=['ot_dsim', 'ot_dsim.bignum_lib'],
    package_dir={'ot_dsim': '.'},
    install_requires=[
        'pycryptodome',
        'tabulate',
    ],
)
