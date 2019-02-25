#!/bin/bash

if [[ ! -z $SONAR_TOKEN ]]; then
  sonar-scanner
fi