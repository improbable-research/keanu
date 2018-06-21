package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorMultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

public class MultivariateGaussianTest {

    @Test
    public void decomp() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{1, 2}, new double[]{1, 2}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.3, 0.3, 0.6}));

        MultivariateGaussian y = new MultivariateGaussian(mu, covarianceMatrix);
        y.sample(new KeanuRandom());
    }

}
