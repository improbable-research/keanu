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

public class GammaVariationalTest {


    @Test
    public void inferHyperParamsFromSamples() {
        // SOURCE OF TRUTH

        Random random = new Random(1);
        double trueTheta = 3.0;
        double trueK = 2.0;

        ConstantDoubleVertex theta = new ConstantDoubleVertex(trueTheta);
        ConstantDoubleVertex k = new ConstantDoubleVertex(trueK);
        ConstantDoubleVertex a = new ConstantDoubleVertex(0.);

        DoubleVertex sourceVertex = new GammaVertex(a, theta, k, random);

        // GENERATE FAKE DATA
        List<Double> doubleValues = getSamples(sourceVertex);

        // SET PRIORS
        DoubleVertex unknownTheta = new SmoothUniformVertex(0.01, 10, random);
        DoubleVertex unknownK = new SmoothUniformVertex(0.01, 10, random);

        // OBSERVE
        new PlateBuilder<Double>()
                .fromIterator(doubleValues.iterator())
                .withFactory((plate, datum) -> {

                    DoubleVertex householdLoad = new GammaVertex(a, unknownTheta, unknownK, random);

                    householdLoad.observe(datum);

                }).build();

        // INFER HYPER PARAMETERS
        doInferenceOn(unknownTheta);

        System.out.println("MAP Theta= " + unknownTheta.getValue());
        System.out.println("MAP K= " + unknownK.getValue());

        assertEquals(trueTheta, unknownTheta.getValue(), 0.1);
        assertEquals(trueK, unknownK.getValue(), 0.1);
    }

    private void doInferenceOn(DoubleVertex unknownVertex) {
        BayesNet inferNet = new BayesNet(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100);

        GradientOptimizer g = new GradientOptimizer(inferNet);

        NonLinearConjugateGradientOptimizer nonLinearConjugateGradientOptimizer = new NonLinearConjugateGradientOptimizer(
                NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
                new SimpleValueChecker(1e-16, 1e-16)
        );

        g.maxAPosteriori(5000, nonLinearConjugateGradientOptimizer);
    }

    private List<Double> getSamples(DoubleVertex knownVertex) {

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < 100000; i++) {
            samples.add(knownVertex.sample());
        }

        return samples;
    }
}

