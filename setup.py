import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='optht',
    version='0.1.0',
    description='Optimal hard threshold for matrix denoising',
    long_description=readme,
    author='N. Benjamin Erichson',
    author_email='nbe@st-andrews.ac.uk',
    url='https://github.com/decarsg/optht',
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    install_requires=['numpy', 'scipy'],
)
