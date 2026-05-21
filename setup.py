from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "GRPO-MA"

setup(
    name="grpo-ma",
    version="0.1.0",
    description="Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO (ICML 2026).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whcpumpkin/GRPO-MA",
    project_urls={
        "Paper": "https://arxiv.org/abs/2509.24494",
        "Source": "https://github.com/whcpumpkin/GRPO-MA",
        "Issues": "https://github.com/whcpumpkin/GRPO-MA/issues",
    },
    author="Hongcheng Wang, Yinuo Huang, Sukai Wang, Guanghui Ren, Hao Dong",
    license="Apache-2.0",
    packages=find_packages(exclude=["scripts", "old", "config", "*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": ["grpo-ma-train=train:main"],
    },
    include_package_data=True,
    zip_safe=False,
)
