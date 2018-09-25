package io.improbable.keanu.algorithms.mcmc.proposal;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class MultivariateGaussianProposalDistributionTest {

    private DoubleTensor mu;
    private DoubleTensor covariance;
    private MultivariateGaussian distribution;
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Before
    public void setUpDistribution() throws Exception {
        mu = DoubleTensor.create(0.0, 1.0);
        covariance = DoubleTensor.create(new double[]{
                1., 2.,
                2., 3.},
            2, 2);

        distribution = MultivariateGaussian.withParameters(mu, covariance);
    }

    @Before
    public void setRandomSeed() throws Exception {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Test
    public void name() {
    }
}