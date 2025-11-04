from setuptools import setup, find_packages

setup(
    name='trapped_atoms_simulation', # Name für pip
    version='0.1.0',
    author='Paul Christ', # (Oder Ihr Name)
    description='Simulationscode für meine Masterarbeit',

    packages=find_packages(),
    
   
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pykeops' 

    ],
)
