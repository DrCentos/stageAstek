import setuptools

REQUIRED_PACKAGES = [
    'google-cloud-storage==1.28.1',
    'pandas==1.5.3',
    'gcsfs==0.7.1',
    'pymssql',
    'beam_mysql',
    'beam_mysql-connector-io',
    'Flask==2.1.0'
]

PACKAGE_NAME = 'TheCodebuzzDataflow'
PACKAGE_VERSION = '0.0.1'

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description='Set up file for the required Dataflow packages ',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
)