package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static io.improbable.keanu.tensor.TensorMatchers.hasShape;
import static io.improbable.keanu.tensor.TensorMatchers.hasValue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class MultinomialVertexTest {

    private IntegerTensor n;
    private DoubleTensor p;

    @Before
    public void prepareParameterTensors() {
        // k = 4
        // N:
        //   1    10
        // 100  1000
        n = IntegerTensor.create(new int[]{1, 10, 100, 1000}, 2, 2);
        // P:
        // 0.     1.
        // .25    0.1
        //
        // 0.     0.
        // .25    0.2
        //
        // 0.     0.
        // .25    0.3
        //
        // 1.     0.
        // .25    0.4
        p = DoubleTensor.create(new double[]{
                0., 1., .25, .1,
                0., 0., .25, .2,
                0., 0., .25, .3,
                1., 0., .25, .4
            },
            4, 2, 2);
        //
    }

    @Test
    public void testSlicing() {
        DoubleTensor t = DoubleTensor.arange(0., 120.).reshape(2, 3, 4, 5);
        assertEquals(0., t.getValue(0, 0, 0, 0), 1e-8);
        assertEquals(60., t.getValue(1, 0, 0, 0), 1e-8);
        assertEquals(20., t.getValue(0, 1, 0, 0), 1e-8);
        assertEquals(5., t.getValue(0, 0, 1, 0), 1e-8);
        assertEquals(1., t.getValue(0, 0, 0, 1), 1e-8);

        assertEquals(DoubleTensor.arange(0., 60.).reshape(3, 4, 5),
            t.slice(0, 0));
        assertEquals(DoubleTensor.arange(60., 120.).reshape(3, 4, 5),
            t.slice(0, 1));

        List<Double> tFlattened = t.asFlatList();
        assertEquals(0., tFlattened.get(0), 1e-8);
        assertEquals(71., tFlattened.get(71), 1e-8);

    }

    @Test
    public void ifTheresOnlyOneValidChoiceItAlwaysReturnsIt() {
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0., 0., 1., 0.).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new int[]{1, 1}, KeanuRandom.getDefaultRandom());
        assertThat(samples, hasValue(0, 0, 100, 0));
    }

    @Test
    public void whenKEquals2ItsBinomial() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8);
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p.transpose());
        DiscreteDistribution binomial = Binomial.withParameters(DoubleTensor.scalar(0.2), n);
        for (int value : ImmutableList.of(1, 2, 9, 10)) {
            DoubleTensor binomialLogProbs = binomial.logProb(IntegerTensor.scalar(value));
            DoubleTensor multinomialLogProbs = multinomial.logProb(IntegerTensor.create(value, 10-value).transpose()).transpose();
            assertThat(multinomialLogProbs, allCloseTo(new Double(1e-6), binomialLogProbs));
        }

    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 10000;
        p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4).transpose();
        n = IntegerTensor.scalar(500);

        MultinomialVertex vertex = new MultinomialVertex(
            new int[] {1, N},
            ConstantVertex.of(n),
            ConstantVertex.of(p)
        );

        IntegerTensor samples = vertex.sample();
        assertThat(samples, hasShape(4, N));

        for (int i = 0; i < samples.getShape()[0]; i++) {
            System.out.println(i);
            IntegerTensor sample = samples.slice(0, i);
            Double probability = p.slice(0, i).scalar();
            double mean = sample.toDouble().average();
            double std = sample.toDouble().standardDeviation();

            double epsilonForMean = 0.5;
            double epsilonForVariance = 5.;
            assertEquals(n.scalar() * probability, mean, epsilonForMean);
            assertEquals(n.scalar() * probability * (1 - probability), std*std, epsilonForVariance);
        }
    }
}
