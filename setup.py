import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pyDIMM",
    version="0.0.2",
    author="Ziqi Rong",
    author_email="rongziqi@sjtu.edu.cn",
    description="A Python Dirichlet Multinomial Mixture Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jck-R/pyDIMM",
    packages=setuptools.find_packages(),
    package_data={
        "": ["clibs/*"],
    },
    # ext_modules=[
    #     setuptools.Extension("pyDIMM.clibs.pyDIMM_libs",["./pyDIMM/clibs/pyDIMM_libs.c"])
    # ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ]
)