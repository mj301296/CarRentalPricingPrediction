from setuptools import setup, find_packages

setup(
    name="car_price_predictor",
    version="0.1.0",
    author="Mrugank Jadhav",
    author_email="mrugankjadhav@gmail.com",
    description="A package to predict car rental prices.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
)
