import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="longtail_ensembles",
    version="0.0.1",
    description="longtail_ensembles",
    long_description="",
    long_description_content_type="test/markdown",
    url="https://github.com/ekellbuch/longtail_ensembles",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=["License :: OSI Approved :: MIT License"],
    python_requires=">=3.7",
)
