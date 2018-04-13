package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class BetaVariationalTest {

    @Test
    public void inferHyperParamsFromSamples() {
        // SOURCE OF TRUTH

        Random random = new Random(1);
        double trueAlpha = 2.0;
        double trueBeta = 2.0;

        ConstantDoubleVertex alpha = new ConstantDoubleVertex(trueAlpha);
        ConstantDoubleVertex beta = new ConstantDoubleVertex(trueBeta);

        DoubleVertex sourceVertex = new BetaVertex(alpha, beta, random);

        // GENERATE FAKE DATA
        List<Double> doubleValues = getSamples(sourceVertex);

        // SET PRIORS
        DoubleVertex unknownAlpha = new SmoothUniformVertex(0.01, 10, random);
        DoubleVertex unknownBeta = new SmoothUniformVertex(0.01, 10, random);

        // OBSERVE
        new PlateBuilder<Double>()
                .fromIterator(doubleValues.iterator())
                .withFactory((plate, datum) -> {

                    DoubleVertex householdLoad = new BetaVertex(unknownAlpha, unknownBeta, random);

                    householdLoad.observe(datum);

                }).build();

        // INFER HYPER PARAMETERS
        doInferenceOn(unknownAlpha);

        System.out.println("MAP Alpha= " + unknownAlpha.getValue());
        System.out.println("MAP Beta= " + unknownBeta.getValue());

        assertEquals(trueAlpha, unknownAlpha.getValue(), 0.1);
        assertEquals(trueBeta, unknownBeta.getValue(), 0.1);
    }

    private void doInferenceOn(DoubleVertex unknownVertex) {
        BayesNet inferNet = new BayesNet(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100);

        GradientOptimizer g = new GradientOptimizer(inferNet);

        NonLinearConjugateGradientOptimizer nonLinearConjugateGradientOptimizer = new NonLinearConjugateGradientOptimizer(
                NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
                new SimpleValueChecker(1e-8, 1e-8)
        );

        g.maxAPosteriori(5000, nonLinearConjugateGradientOptimizer);
    }

    private List<Double> getSamples(DoubleVertex knownVertex) {

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            samples.add(knownVertex.sample());
        }

        return samples;
    }
}
