import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

setuptools.setup(
    name='particletracking',
    version='0.1',
    license=license,
    packages=setuptools.find_packages(
        exclude=('tests', 'docs')
    ),
    url="https://github.com/JamesDownsLab/particletracking",
    install_requires=[
        'opencv-python',
        'tables',
        'pandas',
        'trackpy',
        'dask',
        'fast_histogram',
        'pyclipper',
    ],
    dependency_links=[
        'https://github.com/MikeSmithLabTeam/labvision/tarball/repo/master#egg=package-1.0'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)