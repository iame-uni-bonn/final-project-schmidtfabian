# Revision of ***Forecasting economic activity using a text-classification model***

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/EVOsE4mq)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/schmidtfabian/final_project_schmidtfabian/main.svg)](https://results.pre-commit.ci/latest/github/schmidtfabian/final_project_schmidtfabian/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To run this project, one has to first download
[miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and
[Git](https://git-scm.com/downloads) and then create and activate the environment. To
create and activate the environment first navigate into the root directory of the
repistory and then type the following into the console:

```console
$ mamba env create -f environment.yml
$ conda activate final_project_schmidtfabian
```

To build the project, type

```console
$ pytask
```

If you run into any trouble in doing so go through the steps of "Preparing your system"
and "How to get started on a second machine" on this
[website](https://econ-project-templates.readthedocs.io/en/stable/getting_started/index.html#preparing-your-system)
that belongs to the template that was used to create this project.

To investigate the results of this project, you can then navigate into the `bld` folder
of the repistory that was created after running `pytask` or read the `pdf` that is
located in the root directory of this project. The `src` and the `paper` folder contains
all the source code that is needed to run this project. Files with the prefix `task_`
are `pytask` files and are executed once you type `pytask` into the console. The `tests`
folder contains all the tests that can be run to test the functionality of the defined
functions in the source code. To run all the tests type `pytest` into the console while
having the environment activated. Lastly, the `CHANGES.md` file contains a description
of the main changes of this version of the project compared to the previous version of
the project.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).

***Project by Fabian Schmidt***
