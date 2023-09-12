import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disturbance_CrazyFlie_simulation",
    version="1.0",
    author="Hanyang Hu, Xubo Lyu, Xilun Zhang, Sven Gronauer",
    author_email="hha160@sfu.ca",
    description="Add disturbance to the Reinforcement Learning method in the simulation environments for the CrazyFlie quadrotor. This repository is based on the https://github.com/SvenGronauer/phoenix-drone-simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hu-Hanyang/disturbance-CrazyFile-simulation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=['nose'],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    # install_requires=[
    #     'numpy',
    #     'gym<=0.20.0',
    #     'pybullet',
    #     'torch',
    #     'scipy>= 1.4',
    #     'mpi4py',
    #     'psutil',
    # ],
)