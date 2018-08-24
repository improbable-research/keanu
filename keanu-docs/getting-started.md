## Getting Started

### Simple Example

If you're reading this then you've probably already read the Wikipedia article on
[Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network). If you have
then the classic rain/sprinkler/wet-grass example should be familiar.

```java
public class WetGrass {

    public static void main(String[] args) {

        BoolVertex rain = new Flip(0.2);

        BoolVertex sprinkler = new Flip(
            If.isTrue(rain)
                .then(0.01)
                .orElse(0.4)
        );

        BoolVertex wetGrass = new Flip(
            CPT.of(sprinkler, rain)
                .when(false, false).then(1e-2)
                .when(false, true).then(0.8)
                .when(true, false).then(0.9)
                .orDefault(0.99)
        );

        wetGrass.observe(true);
        
        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            new BayesianNetwork(wetGrass.getConnectedGraph()),
            Arrays.asList(sprinkler, rain),
            100000
        ).drop(10000).downSample(2);

        double probabilityOfRainGivenWetGrass = posteriorSamples.get(rain).probability(isRaining -> isRaining.scalar() == true);

        System.out.println(probabilityOfRainGivenWetGrass);
    }
}
```

### Install

It's recommended that you start with the starter project found [here](https://github.com/improbable-research/keanu-starter).
The starter project is a very simple Keanu project built with gradle. 

To quickly create a new project from the starter project:
```
git clone --branch basic --depth 1 https://github.com/improbable-research/keanu-starter.git
```

If you would like to start using Keanu in an existing project, simply add Keanu as a dependency 
in your gradle or maven build file.

#### Gradle

In your project's build.gradle:

```$groovy
compile group: 'io.improbable', name: 'keanu', version: '0.0.12'
```

#### Maven

In your project's pom.xml:

```
<dependency>
    <groupId>io.improbable</groupId>
    <artifactId>keanu</artifactId>
    <version>0.0.12</version>
</dependency>
```
