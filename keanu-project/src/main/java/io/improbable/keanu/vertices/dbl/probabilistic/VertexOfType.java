package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;

public class VertexOfType {
    private VertexOfType() {
    }

    public static PoissonVertex poisson(Double mu) {
        return poisson(ConstantVertex.of(mu));
    }

    public static PoissonVertex poisson(DoubleVertex mu) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .poisson();
    }

    public static GaussianVertex gaussian(Double mu, Double sigma) {
        return gaussian(ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    public static GaussianVertex gaussian(DoubleVertex mu, DoubleVertex sigma) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.SIGMA, sigma)
            .gaussian();
    }

    public static InverseGammaVertex inverseGamma(double a, double b) {
        return inverseGamma(ConstantVertex.of(a), ConstantVertex.of(b));
    }

    public static InverseGammaVertex inverseGamma(DoubleVertex a, DoubleVertex b) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.A, a)
            .withInput(ParameterName.B, b)
            .inverseGamma();
    }

    public static StudentTVertex studentT(int v) {
        return studentT(ConstantVertex.of(v));
    }

    private static StudentTVertex studentT(ConstantIntegerVertex v) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.V, v)
            .studentT();
    }
}
