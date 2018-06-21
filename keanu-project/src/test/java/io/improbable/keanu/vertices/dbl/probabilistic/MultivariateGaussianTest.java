package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;

public class MultivariateGaussianTest {

    KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void multivariateTest() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{1, 2}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.3, 0.3, 0.6}));

        MultivariateGaussian y = new MultivariateGaussian(mu, covarianceMatrix);
        double sumOne = 0;
        double sumTwo = 0;
        int count = 10000;

        for (int i = 0; i < count; i++) {
            DoubleTensor sample = y.sample(random);
            double foo = sample.getValue(0, 0);
            double bar = sample.getValue(1, 0);

            sumOne += foo;
            sumTwo += bar;

            System.out.println("one: " + foo + ". two: " + bar);
        }

        System.out.println((sumOne / count) + " " + (sumTwo / count));


    }

}
