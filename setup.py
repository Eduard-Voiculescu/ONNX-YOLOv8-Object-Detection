import setuptools
import os

import main

APP_PATH = os.path.dirname(main.__file__)

with open(os.path.join(APP_PATH, 'requirements.txt')) as f:
    _INSTALL_REQUIRES = list(map(lambda s: s.strip(), f.readlines()))

_DESCRIPTION = \
    "ONNX Yolov8 Object Detection"

setuptools.setup(
    name='ONNX-YOLOV8-object-Detection',
    version='0.1.0',
    description=_DESCRIPTION,
    classifiers=[],
    keywords='Onnx',
    author='Eduard Voiculescu',
    author_email='eduardvoiculescu95@gmail.com',
    url='https://github.com/dsoprea/PyInotify',
    license='GPL 2',
    packages=setuptools.find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=_INSTALL_REQUIRES,
)
