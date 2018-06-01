package io.improbable.keanu.vertices.generictensor.probabilistic.discrete;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SelectVertexTest {
    private final Logger log = LoggerFactory.getLogger(SelectVertexTest.class);

    private static double epsilon = 0.01;
    private static int N = 100000;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToOne() {

        LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, new ConstantDoubleTensorVertex(0.25));
        selectableValues.put(TestEnum.B, new ConstantDoubleTensorVertex(0.25));
        selectableValues.put(TestEnum.C, new ConstantDoubleTensorVertex(0.25));
        selectableValues.put(TestEnum.D, new ConstantDoubleTensorVertex(0.25));

        LinkedHashMap<TestEnum, Double> proportions = testSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesNotEquallyWeightedSummingToOne() {

        LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, new ConstantDoubleTensorVertex(0.1));
        selectableValues.put(TestEnum.B, new ConstantDoubleTensorVertex(0.2));
        selectableValues.put(TestEnum.C, new ConstantDoubleTensorVertex(0.3));
        selectableValues.put(TestEnum.D, new ConstantDoubleTensorVertex(0.4));

        LinkedHashMap<TestEnum, Double> proportions = testSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToFour() {

        LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, new ConstantDoubleTensorVertex(1.0));
        selectableValues.put(TestEnum.B, new ConstantDoubleTensorVertex(1.0));
        selectableValues.put(TestEnum.C, new ConstantDoubleTensorVertex(1.0));
        selectableValues.put(TestEnum.D, new ConstantDoubleTensorVertex(1.0));

        LinkedHashMap<TestEnum, Double> proportions = testSample(selectableValues, random);
        LinkedHashMap<TestEnum, DoubleTensorVertex> normalisedSelectableValues = normaliseSelectableValues(selectableValues, 4.0);
        assertProportionsWithinExpectedRanges(normalisedSelectableValues, proportions);
    }

    @Test
    public void fourValuesNotEquallyWeightedSummingToFour() {

        LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, new ConstantDoubleTensorVertex(0.25));
        selectableValues.put(TestEnum.B, new ConstantDoubleTensorVertex(0.75));
        selectableValues.put(TestEnum.C, new ConstantDoubleTensorVertex(1.25));
        selectableValues.put(TestEnum.D, new ConstantDoubleTensorVertex(1.75));

        LinkedHashMap<TestEnum, Double> proportions = testSample(selectableValues, random);
        LinkedHashMap<TestEnum, DoubleTensorVertex> normalisedSelectableValues = normaliseSelectableValues(selectableValues, 4.0);
        assertProportionsWithinExpectedRanges(normalisedSelectableValues, proportions);
    }

    private LinkedHashMap<TestEnum, Double> testSample(LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues,
                                                       KeanuRandom random) {

        SelectVertex<TestEnum> select = new SelectVertex<>(selectableValues);

        LinkedHashMap<TestEnum, Integer> sampleFrequencies = new LinkedHashMap<>();
        sampleFrequencies.put(TestEnum.A, 0);
        sampleFrequencies.put(TestEnum.B, 0);
        sampleFrequencies.put(TestEnum.C, 0);
        sampleFrequencies.put(TestEnum.D, 0);

        for (int i = 0; i < N; i++) {
            TestEnum s = select.sample(random).scalar();
            sampleFrequencies.put(s, sampleFrequencies.get(s) + 1);
        }

        return calculateProportions(sampleFrequencies, N);
    }

    private LinkedHashMap<TestEnum, Double> calculateProportions(LinkedHashMap<TestEnum, Integer> sampleFrequencies, int n) {
        LinkedHashMap<TestEnum, Double> proportions = new LinkedHashMap<>();
        for (Map.Entry<TestEnum, Integer> entry : sampleFrequencies.entrySet()) {
            double proportion = (double) entry.getValue() / n;
            proportions.put(entry.getKey(), proportion);
        }

        return proportions;
    }

    private void assertProportionsWithinExpectedRanges(LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues,
                                                       HashMap<TestEnum, Double> proportions) {

        for (Map.Entry<TestEnum, Double> entry : proportions.entrySet()) {
            log.info(entry.getKey() + ": " + entry.getValue());
            double p = entry.getValue();
            double expected = selectableValues.get(entry.getKey()).getValue().scalar();
            assertEquals(p, expected, epsilon);
        }
    }

    private LinkedHashMap<TestEnum, DoubleTensorVertex> normaliseSelectableValues(LinkedHashMap<TestEnum, DoubleTensorVertex> selectableValues,
                                                                                  double sum) {
        LinkedHashMap<TestEnum, DoubleTensorVertex> normalised = new LinkedHashMap<>();
        for (Map.Entry<TestEnum, DoubleTensorVertex> entry : selectableValues.entrySet()) {
            double normalizedProbability = entry.getValue().getValue().scalar() / sum;
            normalised.put(entry.getKey(), new ConstantDoubleTensorVertex(normalizedProbability));
        }
        return normalised;
    }

    private enum TestEnum {
        A, B, C, D
    }
}