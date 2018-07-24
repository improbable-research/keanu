package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

public class VertexOfType {
    private VertexOfType() {
    }

    public static BetaVertex beta(Double a, Double b) {
        return beta(ConstantVertex.of(a), ConstantVertex.of(b));
    }

    public static BetaVertex beta(DoubleVertex a, DoubleVertex b) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.A, a)
            .withInput(ParameterName.B, b)
            .beta();
    }

    public static ChiSquaredVertex chiSquared(Double k) {
        return chiSquared(ConstantVertex.of(k));
    }

    public static ChiSquaredVertex chiSquared(DoubleVertex k) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.K, k)
            .chiSquared();
    }

    public static ExponentialVertex exponential(Double location, Double lambda) {
        return exponential(ConstantVertex.of(location), ConstantVertex.of(lambda));
    }

    public static ExponentialVertex exponential(DoubleVertex location, DoubleVertex lambda) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.LOCATION, location)
            .withInput(ParameterName.LAMBDA, lambda)
            .exponential();
    }

    public static GammaVertex gamma(Double location, Double theta, Double k) {
        return gamma(ConstantVertex.of(location), ConstantVertex.of(theta), ConstantVertex.of(k));
    }

    public static GammaVertex gamma(DoubleVertex location, DoubleVertex theta, DoubleVertex k) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.LOCATION, location)
            .withInput(ParameterName.THETA, theta)
            .withInput(ParameterName.K, k)
            .gamma();
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


    public static LaplaceVertex laplace(Double mu, Double beta) {
        return laplace(ConstantVertex.of(mu), ConstantVertex.of(beta));
    }

    public static LaplaceVertex laplace(DoubleVertex mu, DoubleVertex beta) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.BETA, beta)
            .laplace();
    }

    public static LogisticVertex logistic(Double mu, Double s) {
        return logistic(ConstantVertex.of(mu), ConstantVertex.of(s));
    }

    public static LogisticVertex logistic(DoubleVertex mu, DoubleVertex s) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.S, s)
            .logistic();
    }

    public static LogNormalVertex logNormal(Double mu, Double sigma) {
        return logNormal(ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    public static LogNormalVertex logNormal(DoubleVertex mu, DoubleVertex sigma) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.SIGMA, sigma)
            .logNormal();
    }

    public static MultivariateGaussian multivariateGaussian(Double mu, Double sigma) {
        return multivariateGaussian(ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    public static MultivariateGaussian multivariateGaussian(DoubleVertex mu, DoubleVertex sigma) {
        return new DistributionVertexBuilder()
            .shaped(mu.getShape())
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.SIGMA, sigma)
            .multivariateGaussian();
    }


    public static PoissonVertex poisson(Double mu) {
        return poisson(ConstantVertex.of(mu));
    }

    public static PoissonVertex poisson(DoubleVertex mu) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MU, mu)
            .poisson();
    }

    public static SmoothUniformVertex smoothUniform(Double min, Double max) {
        return smoothUniform(min, max, 0.01);
    }

    public static SmoothUniformVertex smoothUniform(Double min, Double max, Double sharpness) {
        return smoothUniform(ConstantVertex.of(min), ConstantVertex.of(max), ConstantVertex.of(sharpness));
    }

    public static SmoothUniformVertex smoothUniform(DoubleVertex min, DoubleVertex max, DoubleVertex sharpness) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MIN, min)
            .withInput(ParameterName.MAX, max)
            .withInput(ParameterName.SHARPNESS, sharpness)
            .smoothUniform();
    }

    public static StudentTVertex studentT(int v) {
        return studentT(ConstantVertex.of(v));
    }

    private static StudentTVertex studentT(ConstantIntegerVertex v) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.V, v)
            .studentT();
    }


    public static TriangularVertex triangular(Double min, Double max, Double c) {
        return triangular(ConstantVertex.of(min), ConstantVertex.of(max), ConstantVertex.of(c));
    }

    public static TriangularVertex triangular(DoubleVertex min, DoubleVertex max, DoubleVertex c) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MIN, min)
            .withInput(ParameterName.MAX, max)
            .withInput(ParameterName.C, c)
            .triangular();
    }

    public static UniformVertex uniform(double min, double max) {
        return uniform(ConstantVertex.of(min), ConstantVertex.of(max));
    }

    public static UniformVertex uniform(DoubleVertex min, DoubleVertex max) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MIN, min)
            .withInput(ParameterName.MAX, max)
            .uniform();
    }

    public static UniformIntVertex uniform(int min, int max) {
        return uniform(ConstantVertex.of(min), ConstantVertex.of(max));
    }

    public static UniformIntVertex uniform(IntegerVertex min, IntegerVertex max) {
        return new DistributionVertexBuilder()
            .withInput(ParameterName.MIN, min)
            .withInput(ParameterName.MAX, max)
            .uniformInt();
    }
}
