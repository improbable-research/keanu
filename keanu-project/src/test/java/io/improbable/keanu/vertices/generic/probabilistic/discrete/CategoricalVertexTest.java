package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class CategoricalVertexTest {
    private static double epsilon = 0.01;
    private static int N = 100000;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToOne() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.C, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.D, ConstantVertex.of(0.25));

        Map<TestEnum, Double> proportions = testSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesNotEquallyWeightedSummingToOne() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.2));
        selectableValues.put(TestEnum.C, ConstantVertex.of(0.3));
        selectableValues.put(TestEnum.D, ConstantVertex.of(0.4));

        Map<TestEnum, Double> proportions = testSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToFour() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.B, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.C, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.D, ConstantVertex.of(1.0));

        Map<TestEnum, Double> proportions = testSample(selectableValues, random);
        Map<TestEnum, DoubleVertex> normalisedSelectableValues = normaliseSelectableValues(selectableValues, 4.0);
        assertProportionsWithinExpectedRanges(normalisedSelectableValues, proportions);
    }

    @Test
    public void fourValuesNotEquallyWeightedSummingToFour() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.75));
        selectableValues.put(TestEnum.C, ConstantVertex.of(1.25));
        selectableValues.put(TestEnum.D, ConstantVertex.of(1.75));

        Map<TestEnum, Double> proportions = testSample(selectableValues, random);
        Map<TestEnum, DoubleVertex> normalisedSelectableValues = normaliseSelectableValues(selectableValues, 4.0);
        assertProportionsWithinExpectedRanges(normalisedSelectableValues, proportions);
    }

    @Test
    public void ofDirichletVertexHasCorrectProportions() {
        final DoubleTensor concentration = DoubleTensor.create(1, 2, 3, 4);
        final DirichletVertex dirichletVertex = new DirichletVertex(new ConstantDoubleVertex(concentration));
        final CategoricalVertex<TestEnum> categoricalVertex = CategoricalVertex.of(dirichletVertex, Arrays.asList(TestEnum.A, TestEnum.B, TestEnum.C, TestEnum.D));
        final DoubleTensor sample = dirichletVertex.getValue();

        final Map<TestEnum, DoubleVertex> expectedProportions = new LinkedHashMap<>();
        expectedProportions.put(TestEnum.A, ConstantVertex.of(sample.getValue(0, 0)));
        expectedProportions.put(TestEnum.B, ConstantVertex.of(sample.getValue(0, 1)));
        expectedProportions.put(TestEnum.C, ConstantVertex.of(sample.getValue(0, 2)));
        expectedProportions.put(TestEnum.D, ConstantVertex.of(sample.getValue(0, 3)));

        final Map<TestEnum, Double> proportions = testSampleFromVertex(categoricalVertex, random);
        assertProportionsWithinExpectedRanges(expectedProportions, proportions);
    }

    @Test
    public void ofDirichletVertexUsesIntegerRangeByDefault() {
        final DoubleTensor concentration = DoubleTensor.create(1, 2, 3, 4, 5);
        final DirichletVertex dirichletVertex = new DirichletVertex(new ConstantDoubleVertex(concentration));
        final CategoricalVertex<Integer> categoricalVertex = CategoricalVertex.of(dirichletVertex);
        final DoubleTensor sample = dirichletVertex.getValue();

        final Map<Integer, DoubleVertex> expectedProportions = new LinkedHashMap<>();
        expectedProportions.put(0, ConstantVertex.of(sample.getValue(0, 0)));
        expectedProportions.put(1, ConstantVertex.of(sample.getValue(0, 1)));
        expectedProportions.put(2, ConstantVertex.of(sample.getValue(0, 2)));
        expectedProportions.put(3, ConstantVertex.of(sample.getValue(0, 3)));
        expectedProportions.put(4, ConstantVertex.of(sample.getValue(0, 4)));

        final Map<Integer, Double> proportions = testSampleFromVertex(categoricalVertex, random);
        assertProportionsWithinExpectedRanges(expectedProportions, proportions);
    }

    @Test(expected = IllegalArgumentException.class)
    public void ofDirichletWrongAmountOfCategoriesFails() {
        final DirichletVertex dirichletVertex = new DirichletVertex(1, 2, 3, 4, 5);
        CategoricalVertex.of(dirichletVertex, Arrays.asList(TestEnum.A, TestEnum.B, TestEnum.C, TestEnum.D));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleIfProbabilitiesSumToZero() {
        double probA = 0.0;
        double probB = 0.0;

        LinkedHashMap<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(probA));
        selectableValues.put(TestEnum.B, ConstantVertex.of(probB));

        CategoricalVertex<TestEnum> select = new CategoricalVertex<>(selectableValues);
        select.sample(random);
    }

    @Test
    public void logProbOfCategoryIsEquivalentToItsLogProbabilityDividedBySum() {
        double probA = 0.25;
        double probB = 0.75;
        double probC = 1.25;
        double probD = 1.75;

        double total = probA + probB + probC + probD;

        LinkedHashMap<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.75));
        selectableValues.put(TestEnum.C, ConstantVertex.of(1.25));
        selectableValues.put(TestEnum.D, ConstantVertex.of(1.75));

        CategoricalVertex<TestEnum> select = new CategoricalVertex<>(selectableValues);

        assertEquals(Math.log(probA / total), select.logProb(TestEnum.A), 1e-6);
        assertEquals(Math.log(probB / total), select.logProb(TestEnum.B), 1e-6);
        assertEquals(Math.log(probC / total), select.logProb(TestEnum.C), 1e-6);
        assertEquals(Math.log(probD / total), select.logProb(TestEnum.D), 1e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotComputeLogProbIfProbabilitiesSumToZero() {
        double probA = 0.0;
        double probB = 0.0;

        LinkedHashMap<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(probA));
        selectableValues.put(TestEnum.B, ConstantVertex.of(probB));

        CategoricalVertex<TestEnum> select = new CategoricalVertex<>(selectableValues);
        select.logProb(TestEnum.A);
    }

    private <T> Map<T, Double> testSample(Map<T, DoubleVertex> selectableValues,
                                          KeanuRandom random) {
        return testSampleFromVertex(new CategoricalVertex<>(selectableValues), random);
    }

    private <T> Map<T, Double> testSampleFromVertex(CategoricalVertex<T> vertex, KeanuRandom random) {
        Map<T, Integer> sampleFrequencies = vertex.getSelectableValues().keySet().stream().collect(Collectors.toMap(key -> key, key -> 0));

        for (int i = 0; i < N; i++) {
            T s = vertex.sample(random);
            sampleFrequencies.put(s, sampleFrequencies.get(s) + 1);
        }

        return calculateProportions(sampleFrequencies, N);
    }

    private <T> Map<T, Double> calculateProportions(Map<T, Integer> sampleFrequencies, int n) {
        Map<T, Double> proportions = new LinkedHashMap<>();
        for (Map.Entry<T, Integer> entry : sampleFrequencies.entrySet()) {
            double proportion = (double) entry.getValue() / n;
            proportions.put(entry.getKey(), proportion);
        }

        return proportions;
    }

    private <T> void assertProportionsWithinExpectedRanges(Map<T, DoubleVertex> selectableValues,
                                                           Map<T, Double> proportions) {

        for (Map.Entry<T, Double> entry : proportions.entrySet()) {
            double p = entry.getValue();
            double expected = selectableValues.get(entry.getKey()).getValue().scalar();
            assertEquals(String.format("Sample proportion for category %s is not as expected", entry.getKey()), p, expected, epsilon);
        }
    }

    private <T> Map<T, DoubleVertex> normaliseSelectableValues(Map<T, DoubleVertex> selectableValues,
                                                               double sum) {
        Map<T, DoubleVertex> normalised = new LinkedHashMap<>();
        for (Map.Entry<T, DoubleVertex> entry : selectableValues.entrySet()) {
            double normalizedProbability = entry.getValue().getValue().scalar() / sum;
            normalised.put(entry.getKey(), ConstantVertex.of(normalizedProbability));
        }
        return normalised;
    }

    private enum TestEnum {
        A, B, C, D
    }
}