from setuptools import setup, find_packages

setup(
    name="v2x_tracking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "pyyaml>=5.4.1",
        "matplotlib>=3.4.0",
        "open3d>=0.13.0",
        "numba>=0.53.0",
        "pyquaternion>=0.9.9",
        "tqdm>=4.60.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for Vehicle-Infrastructure Cooperative 3D Tracking",
    keywords="autonomous driving, 3D tracking, V2X, cooperative perception",
    python_requires=">=3.8",
)
