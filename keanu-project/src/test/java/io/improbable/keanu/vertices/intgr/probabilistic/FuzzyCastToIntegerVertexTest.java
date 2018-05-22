package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;

public class FuzzyCastToIntegerVertexTest {

    private final Logger log = LoggerFactory.getLogger(FuzzyCastToIntegerVertexTest.class);

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();
    private KeanuRandom random;

    @Before
    public void setup() {
        this.random = new KeanuRandom(1);
    }

    @Test
    public void evenlySamplesWithUniformInput() {
        int min = 5;
        int max = 15;
        double fuzzinessSigma = 0.0;
        int num = 100000;

        // Input from 4.5 to 15.5 so that there is equal probability of rounding to all integers within range
        DoubleVertex input = new UniformVertex(new ConstantDoubleVertex(4.5), new ConstantDoubleVertex(15.5));
        TreeMap<Integer, Integer> sampleFrequencies = sample(input, fuzzinessSigma, min, max, num);

        log.info("Sample frequencies:");
        printFrequencies(sampleFrequencies);

        double intsInRange = 11.0;
        double expected = 1.0 / intsInRange;

        for (Map.Entry<Integer, Integer> entry : sampleFrequencies.entrySet()) {
            int frequency = entry.getValue();
            double proportion = frequency / (double) num;
            assertEquals(expected, proportion, 0.005);
        }
    }

    @Test
    public void allSamplesBroughtWithinBounds() {
        int min = 5;
        int max = 15;
        double fuzzinessSigma = 0.0;
        int num = 100000;

        DoubleVertex input = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(20.0));
        TreeMap<Integer, Integer> sampleFrequencies = sample(input, fuzzinessSigma, min, max, num);

        log.info("Sample frequencies:");
        printFrequencies(sampleFrequencies);

        assertEquals(sampleFrequencies.firstKey().intValue(), min);
        assertEquals(sampleFrequencies.lastKey().intValue(), max);
    }

    @Test
    public void nearestIntegerIsMostFrequentlySampled() {
        int min = -5;
        int max = 5;
        double fuzzinessSigma = 2.0;
        int num = 100000;

        DoubleVertex input = new ConstantDoubleVertex(0.25);
        TreeMap<Integer, Integer> sampleFrequencies = sample(input, fuzzinessSigma, min, max, num);

        TreeMap<Integer, Integer> sortedByFrequency = new TreeMap<>();

        for (Map.Entry<Integer, Integer> entry : sampleFrequencies.entrySet()) {
            sortedByFrequency.put(entry.getValue(), entry.getKey());
        }

        log.info("Sample frequencies ascending (frequency : value):");
        printFrequencies(sortedByFrequency);

        List<Integer> expectedOrder = Arrays.asList(-5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0);

        int i = 0;
        for (Map.Entry<Integer, Integer> entry : sortedByFrequency.entrySet()) {
            int expectedVal = expectedOrder.get(i++);
            int actualVal = entry.getValue();
            assertEquals(expectedVal, actualVal);
        }
    }

    @Test
    public void onlyNearestIntegerIsSampledWithZeroSigma() {
        int min = -5;
        int max = 5;
        double fuzzinessSigma = 0.0;
        int num = 100000;

        DoubleVertex input = new ConstantDoubleVertex(0.25);
        TreeMap<Integer, Integer> sampleFrequencies = sample(input, fuzzinessSigma, min, max, num);

        log.info("Sample frequencies:");
        printFrequencies(sampleFrequencies);

        assertEquals(num, sampleFrequencies.get(0).intValue());
    }

    @Test
    public void nearestIntegerHasDensityOfOneWithZeroSigma() {
        int min = -5;
        int max = 5;
        double fuzzinessSigma = 0.0;

        DoubleVertex input = new ConstantDoubleVertex(0.25);

        Vertex<Integer> fuzzyCast = new FuzzyCastToIntegerVertex(input, fuzzinessSigma, min, max);
        double density = Math.exp(fuzzyCast.logProbAtValue());

        log.info("Value = " + fuzzyCast.getValue() + ", density = " + density);
        assertEquals(1.0, density, 0.0);
    }

    @Test
    public void calculateMuByObservingFuzzy() {
        DoubleVertex mu = new UniformVertex(0, 10);

        FuzzyCastToIntegerVertex fuzzy = new FuzzyCastToIntegerVertex(mu, 1.0, 0, 10);
        fuzzy.observe(6);

        BayesNetDoubleAsContinuous bayes = new BayesNetDoubleAsContinuous(fuzzy.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayes);
        gradientOptimizer.maxAPosteriori(1000);

        assertEquals(6.0, mu.getValue(), 1e-3);
    }

    @Test
    public void calculate_dP_dmu() {
        double sigma = 1.;
        int min = 1;
        int max = 10;

        double mu1 = 4.0;
        double delta = 0.00001;
        double mu2 = mu1 + delta;
        int observedValue = 5;

        DoubleVertex mu = new UniformVertex(min, max);
        FuzzyCastToIntegerVertex fuzzy = new FuzzyCastToIntegerVertex(mu, sigma, min, max);
        fuzzy.setValue(observedValue);

        mu.setValue(mu1);
        double logDensity1 = fuzzy.logProbAtValue();
        double actual_dPdmu = fuzzy.dLogProbAtValue().get(mu.getId()).scalar();

        mu.setValue(mu2);
        double logDensity2 = fuzzy.logProbAtValue();

        double expected_dPdmu = (logDensity2 - logDensity1) / delta;

        log.info("Expected = " + expected_dPdmu + ", Actual = " + actual_dPdmu);
        assertEquals(expected_dPdmu, actual_dPdmu, 1e-5);
    }

    @Test
    public void calculate_dP_dsigma() {
        IntegerVertex min = new ConstantIntegerVertex(1);
        IntegerVertex max = new ConstantIntegerVertex(10);
        DoubleVertex mu = new ConstantDoubleVertex(5d);
        int observedValue = 5;

        double sigma1 = 2.0;
        double delta = 0.00001;
        double sigma2 = sigma1 + delta;

        DoubleVertex sigma = new UniformVertex(0d, 3d);
        FuzzyCastToIntegerVertex fuzzy = new FuzzyCastToIntegerVertex(mu, sigma, min, max);
        fuzzy.setValue(observedValue);

        sigma.setValue(sigma1);
        double logDensity1 = fuzzy.logProbAtValue();
        double actual_dPdsigma = fuzzy.dLogProbAtValue().get(sigma.getId()).scalar();

        sigma.setValue(sigma2);
        double logDensity2 = fuzzy.logProbAtValue();

        double expected_dPdsigma = (logDensity2 - logDensity1) / delta;

        log.info("Expected: " + expected_dPdsigma + ", Actual: " + actual_dPdsigma);
        assertEquals(expected_dPdsigma, actual_dPdsigma, 1e-5);
    }

    @Test
    public void calculate_dlnP_dmu() {
        double sigma = 1.;
        int min = 1;
        int max = 10;

        double mu1 = 4.0;
        double delta = 0.0001;
        double mu2 = mu1 + delta;
        int observedValue = 5;

        DoubleVertex mu = new UniformVertex(min, max);
        FuzzyCastToIntegerVertex fuzzy = new FuzzyCastToIntegerVertex(mu, sigma, min, max);
        fuzzy.setValue(observedValue);

        mu.setValue(mu1);
        double logDensity1 = fuzzy.logProbAtValue();
        double actual_dlnPdmu = fuzzy.dLogProbAtValue().get(mu.getId()).scalar();

        mu.setValue(mu2);
        double logDensity2 = fuzzy.logProbAtValue();

        double expected_dlnPdmu = (logDensity2 - logDensity1) / delta;

        log.info("Expected = " + expected_dlnPdmu + ", Actual = " + actual_dlnPdmu);
        assertEquals(expected_dlnPdmu, actual_dlnPdmu, 1e-3);
    }

    @Test
    public void calculate_dlnP_dsigma() {
        IntegerVertex min = new ConstantIntegerVertex(1);
        IntegerVertex max = new ConstantIntegerVertex(10);
        DoubleVertex mu = new ConstantDoubleVertex(5d);
        int observedValue = 5;

        double sigma1 = 2.0;
        double delta = 0.0001;
        double sigma2 = sigma1 + delta;

        DoubleVertex sigma = new UniformVertex(0d, 3d);
        FuzzyCastToIntegerVertex fuzzy = new FuzzyCastToIntegerVertex(mu, sigma, min, max);
        fuzzy.setValue(observedValue);

        sigma.setValue(sigma1);
        double logDensity1 = fuzzy.logProbAtValue();
        double actual_dlnPdsigma = fuzzy.dLogProbAtValue().get(sigma.getId()).scalar();

        sigma.setValue(sigma2);
        double logDensity2 = fuzzy.logProbAtValue();

        double expected_dlnPdsigma = (logDensity2 - logDensity1) / delta;

        log.info("Expected: " + expected_dlnPdsigma + ", Actual: " + actual_dlnPdsigma);
        assertEquals(expected_dlnPdsigma, actual_dlnPdsigma, 1e-3);
    }

    private TreeMap<Integer, Integer> sample(DoubleVertex input, double fuzzinessSigma, int min, int max, int num) {

        Vertex<Integer> fuzzyCast = new FuzzyCastToIntegerVertex(input, fuzzinessSigma, min, max);

        TreeMap<Integer, Integer> sampleFrequencies = new TreeMap<>();

        for (int i = 0; i < num; i++) {
            input.setValue(input.sample(random));
            int sample = fuzzyCast.sample(random);
            sampleFrequencies.computeIfAbsent(sample, s -> sampleFrequencies.put(s, 0));
            sampleFrequencies.put(sample, sampleFrequencies.get(sample) + 1);
        }

        return sampleFrequencies;
    }

    private void printFrequencies(TreeMap<Integer, Integer> sampleFrequencies) {

        for (Map.Entry<Integer, Integer> entry : sampleFrequencies.entrySet()) {
            int value = entry.getKey();
            int frequency = entry.getValue();
            log.info(value + " : " + frequency);
        }
    }
}
