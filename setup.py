from setuptools import setup, find_packages
from typing import List

obj = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if obj in requirements:
            requirements.remove(obj)
    return requirements

setup(
    name="GoldPricePrediction",
    version="0.0.1",
    author="Hema_Kalyan",
    author_email="kalyanmurapaka274@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)