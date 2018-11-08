package io.improbable.keanu.vertices.intgr.probabilistic;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.TensorValueException;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.junit.Test;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

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

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheProbabilitiesDontSumToOne() {
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.1, 0.1, 0.1).transpose();
        Multinomial.withParameters(n, p);
    }

    @Test(expected = TensorValueException.class)
    public void inDebugModeItThrowsIfAnyOfTheProbabilitiesIsZero() {
        try {
            Multinomial.CATEGORY_PROBABILITIES_CANNOT_BE_ZERO.enable();
            IntegerTensor n = IntegerTensor.scalar(100);
            DoubleTensor p = DoubleTensor.create(0., 0., 1., 0.).transpose();
            Multinomial.withParameters(n, p);
        } finally {
            Multinomial.CATEGORY_PROBABILITIES_CANNOT_BE_ZERO.disable();
        }
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void itThrowsIfTheParametersAreDifferentShapes() {
        IntegerTensor n = IntegerTensor.create(100, 200);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4).transpose();
        Multinomial.withParameters(n, p);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheSampleShapeDoesntMatchTheShapeOfN() {
        IntegerTensor n = IntegerTensor.create(100, 200);
        DoubleTensor p = DoubleTensor.create(new double[]{
                0.1, 0.25,
                0.2, 0.25,
                0.3, 0.25,
                0.4, 0.25
            },
            4, 2);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        multinomial.sample(new long[]{2, 2}, KeanuRandom.getDefaultRandom());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbShapeDoesntMatchTheNumberOfCategories() {
        IntegerTensor n = IntegerTensor.create(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.scalar(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateDoesntSumToN() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.create(5, 6).transpose());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateContainsNegativeNumbers() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.create(-1, 11).transpose());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateContainsNumbersGreaterThanN() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.3, 0.5).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        int[] state = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 12};
        assertThat(state[0] + state[1] + state[2], equalTo(10));
        multinomial.logProb(IntegerTensor.create(state).transpose());
    }

    @Test
    public void itWorksWithScalars() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(new double[]{0.01, 0.09, 0.9}, 3, 1);
        MultinomialVertex multinomial = new MultinomialVertex(n, ConstantVertex.of(p));
        IntegerTensor samples = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(samples, hasShape(3, 1));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }

    @Test
    public void itWorksWithTensors() {
        IntegerVertex n = ConstantVertex.of(IntegerTensor.create(new int[]{
                1, 5, 8, 10,
                100, 200, 500, 1000},
            2, 4));

        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .2, .3, .8,
                .25, .25, .4, .45,

                .1, .2, .3, .1,
                .50, .25, .4, .45,

                .8, .6, .4, .1,
                .25, .5, .2, .1
            },
            3, 2, 4));
        //
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(3, 2, 4));
        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
                0, 1, 2, 10,
                25, 50, 200, 450,

                0, 0, 2, 0,
                50, 50, 200, 450,

                1, 4, 4, 0,
                25, 100, 100, 100,
            },
            3, 2, 4));
        assertThat(logProb, closeTo(-30.193364297395277, 1e-8));
    }

    @Test
    public void youCanUseAConcatAndReshapeVertexToPipeInTheProbabilities() {
        IntegerVertex n = ConstantVertex.of(IntegerTensor.create(new int[]{
                1, 10,
                100, 1000},
            2, 2));

        DoubleVertex p1 = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .8,
                .25, .2,
            },
            2, 2));

        DoubleVertex p2 = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .1,
                .50, .3,
            },
            2, 2));

        DoubleVertex p3 = ConstantVertex.of(DoubleTensor.create(new double[]{

                .8, .1,
                .25, .5
            },
            2, 2));

        ConcatenationVertex pConcatenated = new ConcatenationVertex(0, p1, p2, p3);
        ReshapeVertex pReshaped = new ReshapeVertex(pConcatenated, 3, 2, 2);
        MultinomialVertex multinomial = new MultinomialVertex(n, pReshaped);
        IntegerTensor sample = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(3, 2, 2));
        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
                0, 10,
                25, 200,

                0, 0,
                50, 300,

                1, 0,
                25, 500,
            },
            3, 2, 2));

        assertThat(logProb, equalTo(-14.165389164658901));
    }


    @Test
    public void youCanSampleWithATensorIfNIsScalarAndPIsAColumnVector() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(4, 1);
        Multinomial multinomial = Multinomial.withParameters(IntegerTensor.scalar(n).reshape(1, 1), p);
        IntegerTensor samples = multinomial.sample(new long[]{2, 2}, KeanuRandom.getDefaultRandom());
        assertThat(samples, hasShape(4, 2, 2));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }

    @Test
    public void ifYourRandomReturnsZeroItSamplesFromTheFirstCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(0.);
        IntegerTensor n = IntegerTensor.scalar(100).reshape(1, 1);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(4, 1);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new long[]{1, 1}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(100, 0, 0, 0));
    }

    @Test
    public void ifYourRandomReturnsOneItSamplesFromTheLastCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(1.);
        IntegerTensor n = IntegerTensor.scalar(100).reshape(1, 1);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).reshape(4, 1);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new long[]{1, 1}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(0, 0, 0, 100));
    }

    @Test
    public void whenKEqualsTwoItsBinomial() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        DiscreteDistribution binomial = Binomial.withParameters(DoubleTensor.scalar(0.2), n);
        for (int value : ImmutableList.of(1, 2, 9, 10)) {
            DoubleTensor binomialLogProbs = binomial.logProb(IntegerTensor.scalar(value));
            DoubleTensor multinomialLogProbs = multinomial.logProb(IntegerTensor.create(value, 10 - value).transpose()).transpose();
            assertThat(multinomialLogProbs, allCloseTo(1e-6, binomialLogProbs));
        }
    }

    enum Color {
        RED, GREEN, BLUE
    }

    @Test
    public void whenKNEqualsOneItsCategorical() {
        IntegerTensor n = IntegerTensor.scalar(1);
        DoubleTensor p = DoubleTensor.create(0.2, .3, 0.5).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);

        Map<Color, DoubleVertex> selectableValues = ImmutableMap.of(
            Color.RED, ConstantVertex.of(p.getValue(0)),
            Color.GREEN, ConstantVertex.of(p.getValue(1)),
            Color.BLUE, ConstantVertex.of(p.getValue(2)));
        CategoricalVertex<Color, GenericTensor<Color>> categoricalVertex = new CategoricalVertex<>(selectableValues);

        double pRed = categoricalVertex.logProb(GenericTensor.scalar(Color.RED));
        assertThat(multinomial.logProb(IntegerTensor.create(1, 0, 0).transpose()).scalar(), closeTo(pRed, 1e-7));
        double pGreen = categoricalVertex.logProb(GenericTensor.scalar(Color.GREEN));
        assertThat(multinomial.logProb(IntegerTensor.create(0, 1, 0).transpose()).scalar(), closeTo(pGreen, 1e-7));
        double pBlue = categoricalVertex.logProb(GenericTensor.scalar(Color.BLUE));
        assertThat(multinomial.logProb(IntegerTensor.create(0, 0, 1).transpose()).scalar(), closeTo(pBlue, 1e-7));
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 10000;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4).reshape(4, 1);
        IntegerTensor n = IntegerTensor.scalar(500).reshape(1, 1);

        MultinomialVertex vertex = new MultinomialVertex(
            new long[]{1, N},
            ConstantVertex.of(n),
            ConstantVertex.of(p)
        );

        IntegerTensor samples = vertex.sample();
        assertThat(samples, hasShape(4, N));

        for (int i = 0; i < samples.getShape()[0]; i++) {
            IntegerTensor sample = samples.slice(0, i);
            Double probability = p.slice(0, i).scalar();
            double mean = sample.toDouble().average();
            double std = sample.toDouble().standardDeviation();

            double epsilonForMean = 0.5;
            double epsilonForVariance = 5.;
            assertEquals(n.scalar() * probability, mean, epsilonForMean);
            assertEquals(n.scalar() * probability * (1 - probability), std * std, epsilonForVariance);
        }
    }
}
