from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ogpr-radar-viewer",
    version="1.0.0",
    author="Heryx",
    author_email="your-email@example.com",
    description="Professional GPR data viewer for OGPR format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Heryx/ogpr-radar-viewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "PyQt6>=6.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "performance": [
            "numba>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ogpr-viewer=ogpr_viewer.main:main",
        ],
    },
)