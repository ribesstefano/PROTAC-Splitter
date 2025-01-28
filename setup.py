import setuptools

setuptools.setup(
    name="protac_splitter",
    version="0.0.1",
    author="Stefano Ribes and Anders Källberg",
    url="https://github.com/ribesstefano/PROTAC-Splitter",
    author_email="ribes.stefano@gmail.com",
    description="A package to split PROTAC SMILES into their substructures.",
    long_description=open("README.md").read(),
    packages=setuptools.find_packages(),
    install_requires=["torch", "scikit-learn", "datasets", "rdkit", "pandas", "joblib", "optuna", "torchmetrics", "transformers", "gradio"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    # package_data={"": ["data/*.h5", "data/*.pkl", "data/*.csv", "models/*.ckpt"]},
)
