#!/bin/bash

./gradlew keanu-python:clean keanu-python:build -i
./gradlew docs:runPythonSnippets -i