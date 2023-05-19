from setuptools import setup, find_packages

setup(
    name="caafe",
    version="0.1",
    packages=find_packages(),
    description="Context-Aware Automated Feature Engineering (CAAFE) is an automated machine learning tool that uses large language models for feature engineering in tabular datasets. It generates Python code for new features along with explanations for their utility, enhancing interpretability.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noah Hollmann, Samuel Müller, Frank Hutter",
    author_email="noah.homa@gmail.com",
    url="https://github.com/automl/CAAFE",
    license="LICENSE.txt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai",
        "kaggle",
        "openml==0.10.0",
        "tabpfn",
    ],
    extras_require={
        "full": ["autofeat", "featuretools", "tabpfn[full]"],
    },
)
