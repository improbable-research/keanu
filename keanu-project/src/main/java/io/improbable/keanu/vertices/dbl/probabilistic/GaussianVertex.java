package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class GaussianVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    private final Random random;

    public GaussianVertex(DoubleVertex mu, DoubleVertex sigma, Random random) {
        this.mu = mu;
        this.sigma = sigma;
        this.random = random;
        setParents(mu, sigma);
    }

    public GaussianVertex(DoubleVertex mu, double sigma, Random random) {
        this(mu, new ConstantDoubleVertex(sigma), random);
    }

    public GaussianVertex(double mu, DoubleVertex sigma, Random random) {
        this(new ConstantDoubleVertex(mu), sigma, random);
    }

    public GaussianVertex(double mu, double sigma, Random random) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma), random);
    }

    public GaussianVertex(DoubleVertex mu, DoubleVertex sigma) {
        this(mu, sigma, new Random());
    }

    public GaussianVertex(double mu, double sigma) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma), new Random());
    }

    public GaussianVertex(double mu, DoubleVertex sigma) {
        this(new ConstantDoubleVertex(mu), sigma, new Random());
    }

    public GaussianVertex(DoubleVertex mu, double sigma) {
        this(mu, new ConstantDoubleVertex(sigma), new Random());
    }


    public DoubleVertex getMu() {
        return mu;
    }

    public DoubleVertex getSigma() {
        return sigma;
    }

    @Override
    public double density(Double value) {
        return Gaussian.pdf(mu.getValue(), sigma.getValue(), value);
    }

    public double logDensity(Double value) {
        return Gaussian.logPdf(mu.getValue(), sigma.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        Gaussian.Diff dP = Gaussian.dPdf(mu.getValue(), sigma.getValue(), getValue());
        return convertDualNumbersToDiff(dP.dPdmu, dP.dPdsigma, dP.dPdx);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        Gaussian.Diff dlnP = Gaussian.dlnPdf(mu.getValue(), sigma.getValue(), getValue());
        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPdsigma, dlnP.dPdx);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPdmu, double dPdsigma, double dPdx) {
        Infinitesimal dPdInputsFromMu = mu.getDualNumber().getInfinitesimal().multiplyBy(dPdmu);
        Infinitesimal dPdInputsFromSigma = sigma.getDualNumber().getInfinitesimal().multiplyBy(dPdsigma);
        Infinitesimal dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);

        return dPdInputs.getInfinitesimals();
    }

    @Override
    public Double sample() {
        return Gaussian.sample(mu.getValue(), sigma.getValue(), random);
    }

}
