from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="salesforce_data_export",
    version="0.1.0",
    author="Jonathan Nelson",
    author_email="jonathan.nelson@ministrybrands.com",
    description="A Streamlit app to export and visualize Salesforce data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:MBGrowthTeam/sfdc-wickedreports.git",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "plotly",
        "simple-salesforce",
        "python-dotenv",
        "pytz",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "salesforce_data_export=salesforce_data_export.app:main",
        ],
    },
    include_package_data=True,  # Include .env file in the package
)
