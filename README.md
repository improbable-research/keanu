```
    __ __                      
   / //_/__  ____ _____  __  __
  / ,< / _ \/ __ `/ __ \/ / / /
 / /| /  __/ /_/ / / / / /_/ / 
/_/ |_\___/\__,_/_/ /_/\__,_/  
```


[![Build Status][travis-image]][travis-url]
[![Quality Gate][sonar-image]][sonar-url]
[![Maven Central][maven-image]][maven-url]
[![Javadocs](https://www.javadoc.io/badge/io.improbable/keanu.svg)](https://www.javadoc.io/doc/io.improbable/keanu)
[![Slack](https://img.shields.io/badge/join%20slack-%23keanu-brightgreen.svg)](https://join.slack.com/t/improbable-eng/shared_invite/enQtMzQ1ODcyMzQ5MjM4LWY5ZWZmNGM2ODc5MmViNmQ3ZTA3ZTY3NzQwOTBlMTkzZmIxZTIxODk0OWU3YjZhNWVlNDU3MDlkZGViZjhkMjc)

## Overview

Keanu is a general purpose probabilistic programming library built in Java and developed by Improbable's research team.
It enables you to build Bayesian networks through which you can make
probabilistic predictions about large, complex and multifaceted problems.

This is an early stage, pre-alpha version of Keanu. We have an ambitious team
attempting to build an even more ambitious product with the help of the open source community.

## Contributing Guidelines
 
We have decided to open source Keanu at such an early stage in order to solicit user feedback 
and help guide the product to success.

Whilst we will always welcome contributions, we would value your time more if it 
were spent applying Keanu to challenging problems and locating its strengths and weaknesses.

Please create a Github Issue if you encounter any bugs or have a feature request.

* Slack: [#keanu](https://join.slack.com/t/improbable-eng/shared_invite/enQtMzQ1ODcyMzQ5MjM4LWY5ZWZmNGM2ODc5MmViNmQ3ZTA3ZTY3NzQwOTBlMTkzZmIxZTIxODk0OWU3YjZhNWVlNDU3MDlkZGViZjhkMjc)
* Issue Tracker: [GitHub Issues](https://github.com/improbable-research/keanu/issues)

## Features

* Probabilistic Programming Operators and Distributions
* Auto-differentiation
* Inference
  * Maximum a posteriori
  * Metropolis Hastings
  * Hamiltonian Monte Carlo
  * Sequential Monte Carlo (Particle Filtering)
* Support for Kotlin
 
## Getting Started

Want to see an example and run it yourself? Head over to [Getting Started](https://improbable-research.github.io/keanu/docs/getting-started).

## Documentation

Want to learn more? Head over to the [Documentation](https://improbable-research.github.io/keanu/).
You can also see our [JavaDocs](https://www.javadoc.io/doc/io.improbable/keanu) or our [Python Docs](https://improbable-research.github.io/keanu/python/latest)

## Future

What does the future entail for Keanu? Find out at [Future](https://improbable-research.github.io/keanu/docs/future).

## Examples

Interested in more technical examples? Explore the examples repo at `/keanu-examples`.

## Development

#### Building the code

We use Gradle, so running `./gradlew build` (or `gradlew.bat build` on Windows) will compile all the code and run all the tests. You can also run the JMH performance benchmarks with `./gradlew runAllBenchmarks`.

#### Annotations

We use [Lombok](https://projectlombok.org/) annotations, which you will need to enable in your IDE.

For IntelliJ:
 - Install the [Lombok plugin](https://plugins.jetbrains.com/plugin/6317-lombok-plugin)
 - Settings > Build, Execution, Deployment > Compiler > Annotation Processors - Enable annotation processing

#### Formatting

We use [Spotless](https://github.com/diffplug/spotless/tree/master/plugin-gradle) and [yapf](https://github.com/google/yapf) to automatically enforce some basic code style checks. If your build fails due to a formatting issue, simply run `./gradlew formatApply` and commit the changes. You need to have run `./gradlew build` or `test` at least once before running `formatApply`, as this installs yapf in your virtual environment.

#### Python Code Generation

[Custom annotations](keanu-project/src/main/java/io/improbable/keanu/annotation) are used for python code generation. It depends on Oracle JDK to build (i.e. does not support open source JDKs). The minimum version requirement is Python 3.6. Simply run `./gradlew codeGen` to generate the code and commit the changes.


## Hiring

Interested in working for Improbable on cool problems? Start [here](https://improbable.io/careers/joining-us)

[travis-image]: https://api.travis-ci.org/improbable-research/keanu.svg?branch=develop
[travis-url]: https://travis-ci.org/improbable-research/keanu
[maven-image]: https://img.shields.io/maven-central/v/io.improbable/keanu.svg?colorB=brightgreen
[maven-url]: https://search.maven.org/artifact/io.improbable/keanu/
[sonar-image]: https://sonarcloud.io/api/project_badges/measure?project=keanu%3Akeanu-project&metric=alert_status
[sonar-url]: https://sonarcloud.io/dashboard?id=keanu%3Akeanu-project
