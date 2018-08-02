"""
2D/3D Bounding Box Library for Computer Vision
"""
from setuptools import find_packages, setup
from os import path
import bbox


dependencies = [
    "numpy>=1.14.5",
    "pyquaternion>=0.9.2"
]

here = path.abspath(path.dirname(__file__))

setup(
    name='bbox',
    version=bbox.__version__,
    url='https://github.com/varunagrawal/bbox',
    license=bbox.__license__,
    author=bbox.__author__,
    author_email=bbox.__email__,
    description='2D/3D bounding box library for Computer Vision',
    long_description=open("README.md", 'r').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
