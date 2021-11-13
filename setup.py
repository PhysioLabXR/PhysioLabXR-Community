import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ReNaLabApp",
    version="0.0.1.dev3",
    author="ApocalyVec",
    author_email="s-vector.lee@hotmail.com",
    description="Reality Navigation Lab App",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ApocalyVec/RealityNavigation",
    project_urls={
        "Bug Tracker": "https://github.com/ApocalyVec/RealityNavigation/issues",
        "Documentation": "https://realitynavigationdocs.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    package_dir={"rena": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)