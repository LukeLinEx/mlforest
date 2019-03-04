from setuptools import setup

setup(
    name='ml_forest',
    packages=['ml_forest'],
    include_package_data=True,
    install_requires=[
        "bson",
        "pymongo",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn>=0.20",
        "jupyter",
        "statsmodels",
        "tornado==4.5.3"
    ]
)
