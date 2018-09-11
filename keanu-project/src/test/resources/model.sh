#!/bin/bash
input=$1
rainFactor=0.1
humidityFactor=2
rain=$(echo "${input}*${rainFactor}" |bc)
humidity=$(echo "${input}*${humidityFactor}" |bc)
echo ${rain} | cat - /Users/georgenash/Documents/G/keanu/keanu-project/src/test/resources/rainOutput.txt > temp && mv temp /Users/georgenash/Documents/G/keanu/keanu-project/src/test/resources/rainOutput.txt
echo ${humidity} | cat - /Users/georgenash/Documents/G/keanu/keanu-project/src/test/resources/humidityOutput.txt > temp && mv temp /Users/georgenash/Documents/G/keanu/keanu-project/src/test/resources/humidityOutput.txt