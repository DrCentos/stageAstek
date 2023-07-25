import setuptools

REQUIRED_PACKAGES = [
    'pillow',
    'google-cloud-storage',
]

setuptools.setup(
    name='nvresize',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
)
