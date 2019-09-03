from setuptools import setup, find_packages

setup(
    name="delete_retrieve_generate",
    description=("Style transfer through a delete, retrieve, and generate framework"),
    url="https://github.com/newknowledge/delete_retrieve_generate.git",
    author="New Knowledge",
    python_requires=">=3.6.0",
    packages=find_packages(),  
    install_requires=[
        "editdistance==0.5.3",
        "joblib==0.13.2",
        "numpy==1.16.4",
        "nltk",
        "Pillow==6.0.0",
        "protobuf==3.8.0",
        "sklearn",
        "six==1.12.0",
        "tensorboardX==1.7",
        "torch==1.1.0",
        "torchvision==0.3.0",
        "pytorch_transformers",
    ],
    include_package_data=True,
)
