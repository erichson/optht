import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='optht',
    version='0.2.0',
    description='Optimal hard threshold for matrix denoising',
    long_description=readme,
    author='N. Benjamin Erichson, Steven Dahdah',
    author_email='erichson@berkeley.edu, Steven.Dahdah@mail.mcgill.ca',
    url='https://github.com/erichson/optht',
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    install_requires=['numpy', 'scipy'],
)
