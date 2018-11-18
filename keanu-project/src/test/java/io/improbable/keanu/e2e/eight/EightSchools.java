package io.improbable.keanu.e2e.eight;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex;

import java.util.Arrays;

public class EightSchools {

    public static void main(String[] args) {
        EightSchools model = new EightSchools();
        model.run();
    }

    public void run() {
        KeanuRandom random = new KeanuRandom();

        int J = 8;
        double[] yObs = new double[]{28., 8., -3., 7., -1., 1., 18., 12.};
        double[] sigma = new double[]{15., 10., 16., 11., 9., 11., 10., 18.};

        DoubleVertex mu = new GaussianVertex(0, 5);
        DoubleVertex tau = new HalfCauchyVertex(5);
        DoubleVertex theta_tilde = new GaussianVertex(new long[]{1, J}, 0, 1);
        DoubleVertex theta = mu.plus(tau.multiply(theta_tilde));
        DoubleVertex y = new GaussianVertex(theta, ConstantVertex.of(sigma));
        y.observe(yObs);

        BayesianNetwork bayesNet = new BayesianNetwork(
            y.getConnectedGraph()
        );

        NUTS sampler = NUTS.builder()
            .adaptEnabled(false)
            .stepSize(0.3)
            .random(random)
            .build();

        NetworkSamples samples = sampler.getPosteriorSamples(
            bayesNet,
            Arrays.asList(tau, mu, theta_tilde, theta),
            6000
        );

        DoubleTensor tau_mean = samples.getDoubleTensorSamples(tau).getAverages();
        DoubleTensor mu_mean = samples.getDoubleTensorSamples(mu).getAverages();
        DoubleTensor theta_mean = samples.getDoubleTensorSamples(theta).getAverages();
        DoubleTensor theta_tilde_mean = samples.getDoubleTensorSamples(theta_tilde).getAverages();

        System.out.println("Average value for tau : " + tau_mean);
        System.out.println("Average value for theta: " + theta_mean);
        System.out.println("Average value for theta_tilde: " + theta_tilde_mean);
        System.out.println("Average value for mu: " + mu_mean);
    }

}
