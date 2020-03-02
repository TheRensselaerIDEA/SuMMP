import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()
	
setuptools.setup(
	name="ExplainableAI-Jose-Figueroa",
	version="0.0.1",
	author="Kristin Bennett, Jose Figueroa, George, Xiao Shao",
	author_email="josefigueroa168@gmail.com",
	description="A collection of explainable AI tools as well as our own 'cadre' model.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/josefigueroa168/ExplainableAI",
	packages=setuptools.find_packages(),
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: MIT License"
	],
	python_requires='>=3.6'
)