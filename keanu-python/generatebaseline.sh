#!/bin/bash

pipenv run pytest --mpl-generate-path=tests/baseline tests/test_sampling.py
