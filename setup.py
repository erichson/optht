import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='optht',
    version='0.2.0',
    description='Optimal hard threshold for matrix denoising',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='N. Benjamin Erichson, Steven Dahdah',
    author_email='erichson@berkeley.edu, Steven.Dahdah@mail.mcgill.ca',
    url='https://github.com/erichson/optht',
    project_urls={
        "Bug Tracker": "https://github.com/erichson/optht/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    install_requires=['numpy', 'scipy'],
    python_requires=">=3.6",
)
