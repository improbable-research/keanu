package io.improbable.keanu.vertices.intgr.probabilistic;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;
import umontreal.ssj.probdistmulti.MultinomialDist;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static io.improbable.keanu.tensor.TensorMatchers.allValues;
import static io.improbable.keanu.tensor.TensorMatchers.hasShape;
import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MultinomialVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Rule
    public ExpectedException thrown = ExpectedException.none();

//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheProbabilitiesDontSumToOne() {
//        IntegerTensor n = IntegerTensor.scalar(100).reshape(1, 1);
//        DoubleTensor p = DoubleTensor.create(0.1, 0.1, 0.1, 0.1).reshape(4, 1);
//        Multinomial.withParameters(n, p);
//    }
//
//    @Test(expected = TensorValueException.class)
//    public void inDebugModeItThrowsIfAnyOfTheProbabilitiesIsZero() {
//        try {
//            Multinomial.CATEGORY_PROBABILITIES_CANNOT_BE_ZERO.enable();
//            IntegerTensor n = IntegerTensor.scalar(100).reshape(1, 1);
//            DoubleTensor p = DoubleTensor.create(0., 0., 1., 0.).reshape(4, 1);
//            Multinomial.withParameters(n, p);
//        } finally {
//            Multinomial.CATEGORY_PROBABILITIES_CANNOT_BE_ZERO.disable();
//        }
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheParametersAreDifferentHighRankShapes() {
//        IntegerTensor n = IntegerTensor.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4);
//        DoubleTensor p = DoubleTensor.linspace(0, 1, 18).reshape(3, 2, 3);
//        Multinomial.withParameters(n, p);
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheParametersAreDifferentShapes() {
//        IntegerTensor n = IntegerTensor.create(1, 2).reshape(1, 2);
//        DoubleTensor p = DoubleTensor.linspace(0, 1, 9).reshape(3, 3);
//        Multinomial.withParameters(n, p);
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheSampleShapeDoesntMatchTheShapeOfN() {
//        IntegerTensor n = IntegerTensor.create(100, 200).reshape(1, 2);
//        DoubleTensor p = DoubleTensor.create(new double[]{
//                0.1, 0.25,
//                0.2, 0.25,
//                0.3, 0.25,
//                0.4, 0.25
//            },
//            4, 2);
//        Multinomial multinomial = Multinomial.withParameters(n, p);
//        multinomial.sample(new long[]{2, 2}, KeanuRandom.getDefaultRandom());
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheLogProbShapeDoesntMatchTheNumberOfCategories() {
//        IntegerTensor n = IntegerTensor.create(100).reshape(1, 1);
//        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(4, 1);
//        Multinomial multinomial = Multinomial.withParameters(n, p);
//        multinomial.logProb(IntegerTensor.scalar(1));
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheLogProbStateDoesntSumToN() {
//        IntegerTensor n = IntegerTensor.scalar(10).reshape(1, 1);
//        DoubleTensor p = DoubleTensor.create(0.2, 0.8).reshape(2, 1);
//        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
//        multinomial.logProb(IntegerTensor.create(5, 6).transpose());
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheLogProbStateContainsNegativeNumbers() {
//        IntegerTensor n = IntegerTensor.scalar(10).reshape(1, 1);
//        DoubleTensor p = DoubleTensor.create(0.2, 0.8).reshape(2, 1);
//        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
//        multinomial.logProb(IntegerTensor.create(-1, 11).transpose());
//    }
//
//    @Test(expected = IllegalArgumentException.class)
//    public void itThrowsIfTheLogProbStateContainsNumbersGreaterThanN() {
//        IntegerTensor n = IntegerTensor.scalar(10).reshape(1, 1);
//        DoubleTensor p = DoubleTensor.create(0.2, 0.3, 0.5).reshape(3, 1);
//        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
//        int[] state = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 12};
//        assertThat(state[0] + state[1] + state[2], equalTo(10));
//        multinomial.logProb(IntegerTensor.create(state).transpose());
//    }

    @Test
    public void itWorksWithScalarTrialCountN() {
        int n = 4;
        DoubleTensor p = DoubleTensor.create(new double[]{0.2, 0.3, 0.5}, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample();

        assertThat(sample, hasShape(3));
        assertThat(sample, allValues(both(greaterThan(-1)).and(lessThan(n))));
        assertThat(sample.sum(), equalTo(n));
    }

    @Test
    public void itWorksWithAlmostOneProbability() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(0.1, new long[]{10});

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor samples = multinomial.sample();

        assertThat(samples, hasShape(10));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }

    @Test
    public void itWorksWithVectorOfNandMatrixOfProbabilities() {

        IntegerTensor n = IntegerTensor.create(new int[]{
            4, 5
        }, 2);

        DoubleTensor p = DoubleTensor.create(new double[]{
            0.1, 0.2, 0.7,
            0.3, 0.3, 0.4
        }, 2, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);

        IntegerTensor sample = multinomial.sample();
        assertThat(sample, hasShape(2, 3));

        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
            1, 1, 2,
            3, 1, 1,
        }, 2, 3));

        MultinomialDist dist1 = new MultinomialDist(4, new double[]{0.1, 0.2, 0.7});
        MultinomialDist dist2 = new MultinomialDist(5, new double[]{0.3, 0.3, 0.4});

        double expected1 = dist1.prob(new int[]{1, 1, 2});
        double expected2 = dist2.prob(new int[]{3, 1, 1});

        double lopSumExpected = Math.log(expected1) + Math.log(expected2);

        assertThat(logProb, closeTo(lopSumExpected, 1e-8));
    }

    @Test
    public void itWorksWithMatrixOfNAndRank3Probabilities() {
        IntegerTensor n = IntegerTensor.create(new int[]{
            1, 10,
            100, 1000
        }, 2, 2);

        DoubleTensor p = DoubleTensor.create(new double[]{
            .1, .1, .8,
            .8, .1, .1,
            .25, .5, .25,
            .2, .3, .5
        }, 2, 2, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(2, 2, 3));

        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
            0, 0, 1,
            10, 0, 0,
            25, 50, 25,
            200, 300, 500
        }, 2, 2, 3));

        MultinomialDist dist1 = new MultinomialDist(1, new double[]{.1, .1, .8});
        MultinomialDist dist2 = new MultinomialDist(10, new double[]{.8, .1, .1});
        MultinomialDist dist3 = new MultinomialDist(100, new double[]{.25, .5, .25});
        MultinomialDist dist4 = new MultinomialDist(1000, new double[]{.2, .3, .5});

        double expected1 = dist1.prob(new int[]{0, 0, 1});
        double expected2 = dist2.prob(new int[]{10, 0, 0});
        double expected3 = dist3.prob(new int[]{25, 50, 25});
        double expected4 = dist4.prob(new int[]{200, 300, 500});

        double expectedLogProb = Math.log(expected1) + Math.log(expected2) + Math.log(expected3) + Math.log(expected4);

        assertThat(logProb, closeTo(expectedLogProb, 1e-8));
    }


    @Test
    public void youCanSampleWithATensorIfNIsScalarAndPIsARowVector() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(1, 4);
        Multinomial multinomial = Multinomial.withParameters(IntegerTensor.scalar(n), p);
        IntegerTensor samples = multinomial.sample(new long[]{1, 4}, KeanuRandom.getDefaultRandom());
        assertThat(samples, hasShape(1, 4));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }

    @Test
    public void ifYourRandomReturnsZeroItSamplesFromTheFirstCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(0.);
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(1, 4);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new long[]{4, 1}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(100, 0, 0, 0));
    }

    @Test
    public void ifYourRandomReturnsOneItSamplesFromTheLastCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(1.);
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(1, 4);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new long[]{1, 4}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(0, 0, 0, 100));
    }

    @Test
    public void whenKEqualsTwoItsBinomial() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).reshape(1, 2);
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        DiscreteDistribution binomial = Binomial.withParameters(DoubleTensor.scalar(0.2), n);

        for (int value : new int[]{1, 2, 9, 10}) {
            DoubleTensor binomialLogProbs = binomial.logProb(IntegerTensor.scalar(value));
            DoubleTensor multinomialLogProbs = multinomial.logProb(IntegerTensor.create(value, 10 - value));
            assertThat(multinomialLogProbs, allCloseTo(1e-6, binomialLogProbs));
        }
    }

    enum Color {
        RED, GREEN, BLUE
    }

    @Test
    public void whenKNEqualsOneItsCategorical() {
        IntegerTensor n = IntegerTensor.scalar(1);
        DoubleTensor p = DoubleTensor.create(0.2, .3, 0.5);
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);

        Map<Color, DoubleVertex> selectableValues = ImmutableMap.of(
            Color.RED, ConstantVertex.of(p.getValue(0)),
            Color.GREEN, ConstantVertex.of(p.getValue(1)),
            Color.BLUE, ConstantVertex.of(p.getValue(2)));
        CategoricalVertex<Color, GenericTensor<Color>> categoricalVertex = new CategoricalVertex<>(selectableValues);

        double pRed = categoricalVertex.logProb(GenericTensor.scalar(Color.RED));
        assertThat(multinomial.logProb(IntegerTensor.create(1, 0, 0)).scalar(), closeTo(pRed, 1e-7));
        double pGreen = categoricalVertex.logProb(GenericTensor.scalar(Color.GREEN));
        assertThat(multinomial.logProb(IntegerTensor.create(0, 1, 0)).scalar(), closeTo(pGreen, 1e-7));
        double pBlue = categoricalVertex.logProb(GenericTensor.scalar(Color.BLUE));
        assertThat(multinomial.logProb(IntegerTensor.create(0, 0, 1)).scalar(), closeTo(pBlue, 1e-7));
    }

    @Category(Slow.class)
    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int sampleCount = 10000;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4).plus(DoubleTensor.zeros(sampleCount, 4));
        int n = 500;

        MultinomialVertex vertex = new MultinomialVertex(n, p);

        IntegerTensor samples = vertex.sample();
        assertThat(samples, hasShape(sampleCount, 4));

        for (int i = 0; i < samples.getShape()[1]; i++) {

            IntegerTensor sampleForIndexI = samples.slice(1, i);

            Double probability = p.getValue(0, i);
            double mean = sampleForIndexI.toDouble().average();
            double std = sampleForIndexI.toDouble().standardDeviation();

            double epsilonForMean = 0.5;
            double epsilonForVariance = 5.;

            assertEquals(n * probability, mean, epsilonForMean);
            assertEquals(n * probability * (1 - probability), std * std, epsilonForVariance);
        }
    }

}