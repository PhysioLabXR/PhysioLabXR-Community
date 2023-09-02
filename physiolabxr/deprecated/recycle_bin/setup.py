import setuptools
from pkg_resources import parse_requirements


requires = [
    'brainflow',
    'numpy',
    'PyQt6',
    'pyserial',
    'pylsl',
    'scikit-learn',
    'scipy~=1.9.1',
    'pyqtgraph',
    'pyxdf',
    'pyscreeze',
    'opencv-python',
    'pyzmq',
    'setuptools',
    'psutil',
    'pytest-qt',
    'numba',
    'PyOpenGL',
    'PyOpenGL_accelerate'
]

setuptools.setup(
    name="ReNaLabApp",
    version="0.0.1.dev13",
    author="Ziheng 'Leo' Li",
    author_email="s-vector.lee@hotmail.com",
    description="Reality Navigation Lab App",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ApocalyVec/RenaLabApp",
    project_urls={
        "Bug Tracker": "https://github.com/ApocalyVec/RenaLabApp/issues",
        "Documentation": "https://realitynavigationdocs.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=requires
)
