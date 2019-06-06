package io.improbable.keanu.vertices.intgr.probabilistic;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
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
import io.improbable.keanu.vertices.intgr.IntegerVertex;
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

    @Test
    public void itThrowsIfTheProbabilitiesDontSumToOne() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Probabilities must sum to 1 but summed to [0.4]");

        int n = 100;
        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(0.1, 0.1, 0.1, 0.1));
        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
        multinomialVertex.setValidationEnabled(true);
        multinomialVertex.sample();
    }

    @Test
    public void inDebugModeItThrowsIfAnyOfTheProbabilitiesIsZero() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Probabilities must be > 0 < 1 but were [0.0, 0.0, 1.0, 0.0]");

        int n = 100;
        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(0, 0, 1, 0));
        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
        multinomialVertex.setValidationEnabled(true);
        multinomialVertex.sample();
    }

    @Test
    public void itThrowsIfTheParametersAreDifferentHighRankShapes() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The shape of n [2, 4] must be broadcastable with the shape of p excluding the k dimension [3, 2]");

        IntegerTensor n = IntegerTensor.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4);
        DoubleTensor p = DoubleTensor.linspace(0, 1, 18).reshape(3, 2, 3);
        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
    }

    @Test
    public void itThrowsIfTheParametersAreDifferentShapes() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The shape of n [2] must be broadcastable with the shape of p excluding the k dimension [3]");

        IntegerTensor n = IntegerTensor.create(1, 2);
        DoubleTensor p = DoubleTensor.linspace(0, 1, 9).reshape(3, 3);
        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
    }

    @Test
    public void doesNotThrowIfNIsBroadcastableToP() {
        IntegerTensor n = IntegerTensor.create(1, 2, 3);
        DoubleTensor p = DoubleTensor.linspace(0, 1, 9).reshape(3, 3);
        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
    }

    @Test
    public void itThrowsIfTheSampleShapeIsNotBroadcastableToTheVertexShape() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage(
            "Shape [2, 2] is incompatible with n shape [4] and p shape [4, 2]." +
                " It must be broadcastable with [4, 2]"
        );

        IntegerTensor n = IntegerTensor.create(100, 200, 300, 400);
        DoubleTensor p = DoubleTensor.create(new double[]{
            0.5, 0.5,
            0.25, 0.75,
            0.3, 0.7,
            0.4, 0.6
        }, 4, 2);

        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
        multinomialVertex.setValidationEnabled(true);
        multinomialVertex.sampleWithShape(new long[]{2, 2});
    }

    @Test
    public void doesAllowSampleWithShapeThatIsBroadcastableWithVertexShape() {
        IntegerTensor n = IntegerTensor.create(100, 200, 300, 400);
        DoubleTensor p = DoubleTensor.create(new double[]{
            0.1, 0.25,
            0.2, 0.25,
            0.3, 0.25,
            0.4, 0.25
        }, 4, 2);

        MultinomialVertex multinomialVertex = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomialVertex.sampleWithShape(new long[]{2, 4, 2}, KeanuRandom.getDefaultRandom());

        assertThat(sample, hasShape(2, 4, 2));
    }

    @Test
    public void itThrowsIfTheLogProbShapeDoesntMatchTheNumberOfCategories() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("x shape must have far right dimension matching number of categories k 4");

        IntegerTensor n = IntegerTensor.create(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4);
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        multinomial.setValidationEnabled(true);
        multinomial.logProb(IntegerTensor.scalar(1));
    }

    @Test
    public void itThrowsIfTheLogProbStateDoesntSumToN() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The sum of x [12] must equal n [10.0]");

        IntegerVertex n = ConstantVertex.of(10);
        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(0.2, 0.8));
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        multinomial.setValidationEnabled(true);
        multinomial.logProb(IntegerTensor.create(5, 7));
    }

    @Test
    public void itAllowsTheLogProbStateToSumToN() {
        IntegerVertex n = ConstantVertex.of(10);
        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(0.2, 0.8));
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        multinomial.setValidationEnabled(true);
        multinomial.logProb(IntegerTensor.create(5, 5, 3, 7).reshape(2, 2));
    }

    @Test
    public void itThrowsIfTheLogProbStateContainsNegativeNumbers() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("x must be >= 0 and <= n");

        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8);
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        multinomial.setValidationEnabled(true);
        multinomial.logProb(IntegerTensor.create(-1, 11).transpose());
    }

    @Test
    public void itThrowsIfTheLogProbStateContainsNumbersGreaterThanN() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("x must be >= 0 and <= n");

        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.3, 0.5);
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        multinomial.setValidationEnabled(true);
        int[] state = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 12};
        assertThat(state[0] + state[1] + state[2], equalTo(10));
        multinomial.logProb(IntegerTensor.create(state).transpose());
    }

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
    public void itWorksWithMatrixOfNAndRank3OfProbabilities() {

        int n1 = 1;
        int n2 = 10;
        int n3 = 100;
        int n4 = 1000;

        IntegerTensor n = IntegerTensor.create(new int[]{
            n1, n2,
            n3, n4
        }, 2, 2);

        double[] p1 = new double[]{.3, .2, .5};
        double[] p2 = new double[]{.5, .3, .2};
        double[] p3 = new double[]{.2, .2, .6};
        double[] p4 = new double[]{.6, .2, .2};

        DoubleTensor p = DoubleTensor.create(Doubles.concat(
            p1, p2, p3, p4
        ), 2, 2, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample();
        assertThat(sample, hasShape(2, 2, 3));

        int[] x1 = new int[]{0, 0, 1};
        int[] x2 = new int[]{10, 0, 0};
        int[] x3 = new int[]{25, 50, 25};
        int[] x4 = new int[]{200, 300, 500};

        double logProb = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1, x2, x3, x4
        ), 2, 2, 3));

        MultinomialDist dist1 = new MultinomialDist(n1, p1);
        double expected1 = dist1.prob(x1);

        MultinomialDist dist2 = new MultinomialDist(n2, p2);
        double expected2 = dist2.prob(x2);

        MultinomialDist dist3 = new MultinomialDist(n3, p3);
        double expected3 = dist3.prob(x3);

        MultinomialDist dist4 = new MultinomialDist(n4, p4);
        double expected4 = dist4.prob(x4);

        double expectedLogProb = Math.log(expected1) + Math.log(expected2) + Math.log(expected3) + Math.log(expected4);
        assertThat(logProb, closeTo(expectedLogProb, 1e-8));

        int[] x5 = new int[]{0, 1, 0};
        int[] x6 = new int[]{0, 10, 0};
        int[] x7 = new int[]{50, 25, 25};
        int[] x8 = new int[]{500, 200, 300};

        double logProb2x = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1, x2, x3, x4,
            x5, x6, x7, x8
        ), 2, 2, 2, 3));

        MultinomialDist dist5 = new MultinomialDist(n1, p1);
        double expected5 = dist5.prob(x5);

        MultinomialDist dist6 = new MultinomialDist(n2, p2);
        double expected6 = dist6.prob(x6);

        MultinomialDist dist7 = new MultinomialDist(n3, p3);
        double expected7 = dist7.prob(x7);

        MultinomialDist dist8 = new MultinomialDist(n4, p4);
        double expected8 = dist8.prob(x8);

        double expectedLogProb2x = expectedLogProb + Math.log(expected5) +
            Math.log(expected6) + Math.log(expected7) + Math.log(expected8);

        assertThat(logProb2x, closeTo(expectedLogProb2x, 1e-8));
    }

    @Test
    public void itWorksWithVectorOfNAndMatrixOfProbabilities() {

        int n1 = 1;
        int n2 = 10;

        IntegerTensor n = IntegerTensor.create(new int[]{
            n1, n2
        }, 2);

        double[] p1 = new double[]{.3, .2, .5};
        double[] p2 = new double[]{.5, .3, .2};

        DoubleTensor p = DoubleTensor.create(Doubles.concat(
            p1,
            p2
        ), 2, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample();
        assertThat(sample, hasShape(2, 3));

        int[] x1 = new int[]{0, 0, 1};
        int[] x2 = new int[]{10, 0, 0};

        double logProb = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1,
            x2
        ), 2, 3));

        MultinomialDist dist1 = new MultinomialDist(n1, p1);
        double expected1 = dist1.prob(x1);

        MultinomialDist dist2 = new MultinomialDist(n2, p2);
        double expected2 = dist2.prob(x2);

        double expectedLogProb = Math.log(expected1) + Math.log(expected2);
        assertThat(logProb, closeTo(expectedLogProb, 1e-8));

        int[] x3 = new int[]{0, 1, 0};
        int[] x4 = new int[]{0, 10, 0};

        double logProb2x = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1, x2,
            x3, x4
        ), 2, 2, 3));

        MultinomialDist dist3 = new MultinomialDist(n1, p1);
        double expected3 = dist3.prob(x3);

        MultinomialDist dist4 = new MultinomialDist(n2, p2);
        double expected4 = dist4.prob(x4);

        double expectedLogProb2x = expectedLogProb + Math.log(expected3) + Math.log(expected4);

        assertThat(logProb2x, closeTo(expectedLogProb2x, 1e-8));
    }

    @Test
    public void itWorksWithMatrixOfNAndMatrixOfProbabilities() {

        int n1 = 1;
        int n2 = 10;
        int n3 = 100;
        int n4 = 1000;

        IntegerTensor n = IntegerTensor.create(new int[]{
            n1, n2,
            n3, n4
        }, 2, 2);

        double[] p1 = new double[]{.3, .2, .5};
        double[] p2 = new double[]{.5, .3, .2};

        DoubleTensor p = DoubleTensor.create(Doubles.concat(
            p1,
            p2
        ), 2, 3);

        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample();
        assertThat(sample, hasShape(2, 2, 3));

        int[] x1 = new int[]{0, 0, 1};
        int[] x2 = new int[]{10, 0, 0};
        int[] x3 = new int[]{25, 50, 25};
        int[] x4 = new int[]{200, 300, 500};

        double logProb = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1,
            x2,
            x3,
            x4
        ), 2, 2, 3));

        MultinomialDist dist1 = new MultinomialDist(n1, p1);
        double expected1 = dist1.prob(x1);

        MultinomialDist dist2 = new MultinomialDist(n2, p2);
        double expected2 = dist2.prob(x2);

        MultinomialDist dist3 = new MultinomialDist(n3, p1);
        double expected3 = dist3.prob(x3);

        MultinomialDist dist4 = new MultinomialDist(n4, p2);
        double expected4 = dist4.prob(x4);

        double expectedLogProb = Math.log(expected1) + Math.log(expected2) + Math.log(expected3) + Math.log(expected4);
        assertThat(logProb, closeTo(expectedLogProb, 1e-8));

        int[] x5 = new int[]{0, 1, 0};
        int[] x6 = new int[]{0, 10, 0};
        int[] x7 = new int[]{50, 25, 25};
        int[] x8 = new int[]{500, 200, 300};

        double logProb2x = multinomial.logProb(IntegerTensor.create(Ints.concat(
            x1, x2, x3, x4,
            x5, x6, x7, x8
        ), 2, 2, 2, 3));

        MultinomialDist dist5 = new MultinomialDist(n1, p1);
        double expected5 = dist5.prob(x5);

        MultinomialDist dist6 = new MultinomialDist(n2, p2);
        double expected6 = dist6.prob(x6);

        MultinomialDist dist7 = new MultinomialDist(n3, p1);
        double expected7 = dist7.prob(x7);

        MultinomialDist dist8 = new MultinomialDist(n4, p2);
        double expected8 = dist8.prob(x8);

        double expectedLogProb2x = expectedLogProb + Math.log(expected5) +
            Math.log(expected6) + Math.log(expected7) + Math.log(expected8);

        assertThat(logProb2x, closeTo(expectedLogProb2x, 1e-8));
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
        CategoricalVertex<Color> categoricalVertex = new CategoricalVertex<>(selectableValues);

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
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4);
        int n = 500;

        MultinomialVertex vertex = new MultinomialVertex(n, p);

        IntegerTensor samples = vertex.sampleWithShape(new long[]{sampleCount, 4});
        assertThat(samples, hasShape(sampleCount, 4));

        for (int i = 0; i < samples.getShape()[1]; i++) {

            IntegerTensor sampleForIndexI = samples.slice(1, i);

            Double probability = p.getValue(i);
            double mean = sampleForIndexI.toDouble().average();
            double std = sampleForIndexI.toDouble().standardDeviation();

            double epsilonForMean = 0.5;
            double epsilonForVariance = 5.;

            assertEquals(n * probability, mean, epsilonForMean);
            assertEquals(n * probability * (1 - probability), std * std, epsilonForVariance);
        }
    }

}