from setuptools import setup, find_packages

setup(
    name='ot_dsim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pycryptodome',
        'tabulate',
    ],
)
