package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class LaplaceVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex beta;
    private final Random random;

    public LaplaceVertex(DoubleVertex mu, DoubleVertex beta, Random random) {
        this.mu = mu;
        this.beta = beta;
        this.random = random;
        setParents(mu, beta);
    }

    public LaplaceVertex(DoubleVertex mu, double beta, Random random) {
        this(mu, new ConstantDoubleVertex(beta), random);
    }

    public LaplaceVertex(double mu, DoubleVertex beta, Random random) {
        this(new ConstantDoubleVertex(mu), beta, random);
    }

    public LaplaceVertex(double mu, double beta, Random random) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta), random);
    }

    public LaplaceVertex(DoubleVertex mu, DoubleVertex beta) {
        this(mu, beta, new Random());
    }

    public LaplaceVertex(DoubleVertex mu, double beta) {
        this(mu, new ConstantDoubleVertex(beta), new Random());
    }

    public LaplaceVertex(double mu, DoubleVertex beta) {
        this(new ConstantDoubleVertex(mu), beta, new Random());
    }

    public LaplaceVertex(double mu, double beta) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta), new Random());
    }

    @Override
    public double density(Double value) {
        return Laplace.pdf(mu.getValue(), beta.getValue(), value);
    }

    public double logDensity(Double value) {
        return Laplace.logPdf(mu.getValue(), beta.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        Laplace.Diff diff = Laplace.dPdf(mu.getValue(), beta.getValue(), getValue());
        return convertDualNumbersToDiff(diff.dPdmu, diff.dPdbeta, diff.dPdx);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        Laplace.Diff diff = Laplace.dlnPdf(mu.getValue(), beta.getValue(), getValue());
        return convertDualNumbersToDiff(diff.dPdmu, diff.dPdbeta, diff.dPdx);
    }

    @Override
    public Double sample() {
        return Laplace.sample(mu.getValue(), beta.getValue(), random);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPdmu, double dPdbeta, double dPdx) {
        Infinitesimal dPdInputsFromMu = mu.getDualNumber().getInfinitesimal().multiplyBy(dPdmu);
        Infinitesimal dPdInputsFromSigma = beta.getDualNumber().getInfinitesimal().multiplyBy(dPdbeta);
        Infinitesimal dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);

        return dPdInputs.getInfinitesimals();
    }
}
