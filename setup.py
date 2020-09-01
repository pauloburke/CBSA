import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cbsa",
    version="0.0.4",
    author="Paulo Burke",
    author_email="pauloepburke@gmail.com",
    description="A Python package to simulate biochemical systems using the Constraint-Based Simulation Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pauloburke/CBSA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy'],
)
