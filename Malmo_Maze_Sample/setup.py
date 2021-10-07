import setuptools

setuptools.setup(
    name="malmo_maze_env",
    version="0.0.1",
    author="Tsuyoshi Matsuzaki",
    description="A gym environemnt for Malmo",
    url="https://github.com/tsmatz/aaaaaaaaaaaaaaaa",
    install_requires=['gym', 'numpy', 'malmo'],
    packages=setuptools.find_packages(),
    package_data={'custom_malmo_env': ['mission_files/*.xml', 'shell_files/*.sh']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
