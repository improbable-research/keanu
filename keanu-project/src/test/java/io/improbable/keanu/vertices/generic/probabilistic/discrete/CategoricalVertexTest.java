package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import static org.hamcrest.Matchers.lessThan;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import org.junit.rules.ExpectedException;

public class CategoricalVertexTest {
    private static double epsilon = 0.01;
    private static int N = 100000;

    private KeanuRandom random;

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void cantCreateCategoricalVertexIfShapeIsNotSpecifiedAndNonScalarShapesDoNotMatch() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);
        DoubleTensor t2 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 4, 1);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(t2));

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Shapes must match or be scalar");

        new CategoricalVertex<>(selectableValues);
    }

    @Test
    public void cantCreateCategoricalVertexIfShapeIsNotSpecifiedAndNonScalarLengthsDoNotMatch() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5}, 2, 1);
        DoubleTensor t2 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 4, 1);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(t2));

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Shapes must match or be scalar");

        new CategoricalVertex<>(selectableValues);
    }

    @Test
    public void canCreateCategoricalVertexIfShapeIsNotSpecifiedAndNonScalarShapesMatch() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);
        DoubleTensor t2 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(t2));

        new CategoricalVertex<>(selectableValues);
    }

    @Test
    public void canCreateCategoricalVertexIfShapeIsNotSpecifiedAndAllShapesAreScalar() {
        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.5));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.8));

        new CategoricalVertex<>(selectableValues);
    }

    @Test
    public void cantCreateCategoricalVertexIfNonScalarShapeDoNotMatchProposedShape() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));

        int[] proposalShape = new int[]{3, 5, 6};

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Proposed shape " + Arrays.toString(proposalShape) + " does not match other non scalar shapes");

        new CategoricalVertex<>(proposalShape, selectableValues);
    }

    @Test
    public void cantCreateCategoricalVertexIfShapeIsSpecifiedAndNonScalarShapeDoNotMatch() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);
        DoubleTensor t2 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 4, 2);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(t2));

        int[] proposalShape = new int[]{3, 5, 6};

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("More than a single non-scalar shape");

        new CategoricalVertex<>(proposalShape, selectableValues);
    }

    @Test
    public void canCreateCategoricalVertexIfShapeIsSpecifiedAndNonScalarShapeMatchProposalShapeOrIsScalar() {
        DoubleTensor t1 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);
        DoubleTensor t2 = DoubleTensor.create(new double[] {0., 0.5, 0.8, 0.2}, 2, 2);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(t1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(t2));
        selectableValues.put(TestEnum.C, ConstantVertex.of(1.));

        int[] proposalShape = new int[]{2, 2};

        new CategoricalVertex<>(proposalShape, selectableValues);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToOne() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.C, ConstantVertex.of(0.25));
        selectableValues.put(TestEnum.D, ConstantVertex.of(0.25));

        Map<TestEnum, Double> proportions = testScalarSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesNotEquallyWeightedSummingToOne() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.1));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.2));
        selectableValues.put(TestEnum.C, ConstantVertex.of(0.3));
        selectableValues.put(TestEnum.D, ConstantVertex.of(0.4));

        Map<TestEnum, Double> proportions = testScalarSample(selectableValues, random);
        assertProportionsWithinExpectedRanges(selectableValues, proportions);
    }

    @Test
    public void fourValuesEquallyWeightedSummingToFour() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.B, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.C, ConstantVertex.of(1.0));
        selectableValues.put(TestEnum.D, ConstantVertex.of(1.0));

        Map<TestEnum, Double> proportions = testScalarSample(selectableValues, random);
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

        Map<TestEnum, Double> proportions = testScalarSample(selectableValues, random);
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

        final Map<TestEnum, Double> proportions = testScalarSampleFromVertex(categoricalVertex, random);
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

        final Map<Integer, Double> proportions = testScalarSampleFromVertex(categoricalVertex, random);
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

        assertEquals(Math.log(probA / total), select.logProb(new GenericTensor<>(TestEnum.A)), 1e-6);
        assertEquals(Math.log(probB / total), select.logProb(new GenericTensor<>(TestEnum.B)), 1e-6);
        assertEquals(Math.log(probC / total), select.logProb(new GenericTensor<>(TestEnum.C)), 1e-6);
        assertEquals(Math.log(probD / total), select.logProb(new GenericTensor<>(TestEnum.D)), 1e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotComputeLogProbIfProbabilitiesSumToZero() {
        double probA = 0.0;
        double probB = 0.0;

        LinkedHashMap<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(probA));
        selectableValues.put(TestEnum.B, ConstantVertex.of(probB));

        CategoricalVertex<TestEnum> select = new CategoricalVertex<>(selectableValues);
        select.logProb(new GenericTensor<>(TestEnum.A));
    }

    @Test
    public void canObserveAGenericTensorAndInferItsProbabilities() {

        UniformVertex A = new UniformVertex(0., 1.);
        UniformVertex B = new UniformVertex(0., 1.);

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, A);
        selectableValues.put(TestEnum.B, B);
        CategoricalVertex<TestEnum> categoricalVertex = new CategoricalVertex<>(selectableValues);

        GenericTensor<TestEnum> observation = new GenericTensor<>(new TestEnum[]{TestEnum.A, TestEnum.A, TestEnum.A, TestEnum.A}, 2, 2);
        categoricalVertex.observe(observation);

        DoubleTensor sampleA = A.sample();
        DoubleTensor sampleB = B.sample();

        assertThat(sampleB.scalar(), lessThan(sampleA.scalar()));
    }

    @Test
    public void canComputeLogProbOfGenericTensor() {
        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();

        double[] chosenValues = {0.25, 0.1, 0.6, 0.5};

        DoubleTensor probA = DoubleTensor.create(new double[]{chosenValues[0], chosenValues[1], 0.4, 0.5}, 2, 2);
        selectableValues.put(TestEnum.A, ConstantVertex.of(probA));
        selectableValues.put(TestEnum.B, ConstantVertex.of(DoubleTensor.create(new double[]{0.75, 0.9, chosenValues[2], chosenValues[3]}, 2, 2)));
        CategoricalVertex<TestEnum> categoricalVertex = new CategoricalVertex<>(selectableValues);

        GenericTensor<TestEnum> value = new GenericTensor<>(new TestEnum[]{TestEnum.A, TestEnum.A, TestEnum.B, TestEnum.B}, 2, 2);
        double logProbA = categoricalVertex.logProb(value);
        double expectedLogProb = Arrays.stream(chosenValues).map(Math::log).sum();

        assertThat(expectedLogProb, closeTo(logProbA, 1e-6));
    }

    private <T> Map<T, Double> testScalarSample(Map<T, DoubleVertex> selectableValues,
                                                KeanuRandom random) {
        return testScalarSampleFromVertex(new CategoricalVertex<>(selectableValues), random);
    }

    private <T> Map<T, Double> testScalarSampleFromVertex(CategoricalVertex<T> vertex, KeanuRandom random) {
        Map<T, Integer> sampleFrequencies = vertex.getSelectableValues().keySet().stream().collect(Collectors.toMap(key -> key, key -> 0));

        for (int i = 0; i < N; i++) {
            T s = vertex.sample(random).scalar();
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