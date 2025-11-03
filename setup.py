from setuptools import setup, find_packages

setup(
    name='trapped_atoms_simulation', # Name für pip
    version='0.1.0',
    author='Paul Christ', # (Oder Ihr Name)
    description='Simulationscode für meine Masterarbeit',
    
    # Diese Zeile ist entscheidend:
    # Sie findet automatisch alle Ordner mit einer __init__.py
    packages=find_packages(),
    
    # Fügen Sie hier alle Pakete hinzu, die Ihr Code braucht
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pykeops' 
        # (Alle 'import ...' am Anfang Ihrer Skripte)
    ],
)
