from setuptools import setup, find_packages

setup(
    name="pbc",
    version="0.1.0",
    description="Pixel Block Chain - Steganographic Image Provenance & Integrity Protocol",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="François Légaré",
    author_email="flegare@gmail.com",
    url="https://github.com/flegare/pixel-block-chain",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "Pillow>=9.0",
    ],
    entry_points={
        "console_scripts": [
            "pbc=pbc.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Security :: Cryptography",
    ],
)
