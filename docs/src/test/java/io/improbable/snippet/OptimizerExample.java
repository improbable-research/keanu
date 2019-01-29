package io.improbable.snippet;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ConjugateGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.BOBYQA;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.OptimizerBounds;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OptimizerExample {

    private static final Logger log = LoggerFactory.getLogger(OptimizerExample.class);

    public static void main(String[] args) {

        UniformVertex temperature = new UniformVertex(20., 30.);
        GaussianVertex firstThermometer = new GaussianVertex(temperature, 2.5);
        GaussianVertex secondThermometer = new GaussianVertex(temperature, 5.);
        firstThermometer.observe(25.);

        log.info("Gradient Optimizer, temperature: " + runGradientOptimizer(temperature));
        log.info("Non-Gradient Optimizer, temperature: " + runNonGradientOptimizer(temperature));

    }

    private static double runGradientOptimizer(DoubleVertex temperature) {
        //%%SNIPPET_START%% GradientOptimizerMostProbable
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.builderFor(temperature.getConnectedGraph())
            .algorithm(ConjugateGradient.builder()
                .maxEvaluations(5000)
                .relativeThreshold(1e-8)
                .absoluteThreshold(1e-8)
                .build())
            .build();
        optimizer.maxAPosteriori();

        double calculatedTemperature = temperature.getValue().scalar();
        //%%SNIPPET_END%% GradientOptimizerMostProbable

        return calculatedTemperature;
    }

    private static double runNonGradientOptimizer(DoubleVertex temperature) {
        //%%SNIPPET_START%% NonGradientOptimizerMostProbable
        OptimizerBounds temperatureBounds = new OptimizerBounds().addBound(temperature.getId(), -250., 250.0);
        NonGradientOptimizer optimizer = KeanuOptimizer.NonGradient.builderFor(temperature.getConnectedGraph())
            .algorithm(BOBYQA.builder()
                .maxEvaluations(5000)
                .boundsRange(100000)
                .optimizerBounds(temperatureBounds)
                .initialTrustRegionRadius(5.)
                .stoppingTrustRegionRadius(2e-8)
                .build())
            .build();
        optimizer.maxAPosteriori();

        double calculatedTemperature = temperature.getValue().scalar();
        //%%SNIPPET_END%% NonGradientOptimizerMostProbable

        return calculatedTemperature;
    }
}
