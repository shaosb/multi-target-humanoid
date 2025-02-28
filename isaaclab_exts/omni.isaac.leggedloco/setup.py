"""Installation script for the 'omni.isaac.leggedloco' python package."""


from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "scipy>=1.7.1",
    # RL
    "torch>=1.9.0",
]

# Installation operation
setup(
    name="omni-isaac-leggedloco",
    version="0.0.1",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=["omni.isaac.leggedloco"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF
