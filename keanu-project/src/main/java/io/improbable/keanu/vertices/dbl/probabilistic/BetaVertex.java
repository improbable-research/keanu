package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class BetaVertex extends ProbabilisticDouble {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;
    private final Random random;

    public BetaVertex(DoubleVertex alpha, DoubleVertex beta, Random random) {
        this.alpha = alpha;
        this.beta = beta;
        this.random = random;
        setParents(alpha, beta);
    }

    public BetaVertex(DoubleVertex alpha, double beta, Random random) {
        this(alpha, new ConstantDoubleVertex(beta), random);
    }

    public BetaVertex(double alpha, DoubleVertex beta, Random random) {
        this(new ConstantDoubleVertex(alpha), beta, random);
    }

    public BetaVertex(double alpha, double beta, Random random) {
        this(new ConstantDoubleVertex(alpha), beta, random);
    }

    public BetaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this(alpha, beta, new Random());
    }

    public BetaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta), new Random());
    }

    public BetaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta, new Random());
    }

    public BetaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), beta, new Random());
    }

    @Override
    public double density(Double value) {
        return Beta.pdf(alpha.getValue(), beta.getValue(), value);
    }

    @Override
    public double logDensity(Double value) {
        return Beta.logPdf(alpha.getValue(), beta.getValue(), value);
    }

    public DoubleVertex getAlpha() {
        return alpha;
    }

    public DoubleVertex getBeta() {
        return beta;
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        Beta.Diff dPdf = Beta.dPdf(alpha.getValue(), beta.getValue(), getValue());
        return convertDualNumbersToDiff(dPdf.dPdAlpha, dPdf.dPdBeta, dPdf.dPdx);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        Beta.Diff dlnPdf = Beta.dlnPdf(alpha.getValue(), beta.getValue(), getValue());
        return convertDualNumbersToDiff(dlnPdf.dPdAlpha, dlnPdf.dPdBeta, dlnPdf.dPdx);
    }

    public Map<String, Double> convertDualNumbersToDiff(double dPdAlpha, double dPdBeta, double dPdx) {
        Infinitesimal dPdInputsFromAlpha = alpha.getDualNumber().getInfinitesimal().multiplyBy(dPdAlpha);
        Infinitesimal dPdInputsFromBeta = beta.getDualNumber().getInfinitesimal().multiplyBy(dPdBeta);
        Infinitesimal dPdInputs = dPdInputsFromAlpha.add(dPdInputsFromBeta);
        dPdInputs.getInfinitesimals().put(getId(), dPdx);

        return dPdInputs.getInfinitesimals();
    }

    @Override
    public Double sample() {
        return Beta.sample(alpha.getValue(), beta.getValue(), 0, 1, random);
    }

}
