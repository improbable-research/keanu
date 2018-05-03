package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;
import java.util.Random;

public class GammaVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex theta;
    private final DoubleVertex k;
    private final Random random;

    /**
     * @param a      location
     * @param theta  scale
     * @param k      shape
     * @param random source for sampling
     */
    public GammaVertex(DoubleVertex a, DoubleVertex theta, DoubleVertex k, Random random) {
        this.a = a;
        this.theta = theta;
        this.k = k;
        this.random = random;
        setParents(a, theta, k);
    }

    public GammaVertex(DoubleVertex theta, double k, Random random) {
        this(new ConstantDoubleVertex(0.0), theta, new ConstantDoubleVertex(k), random);
    }

    public GammaVertex(double theta, DoubleVertex k, Random random) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), k, random);
    }

    public GammaVertex(double theta, double k, Random random) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k), random);
    }

    public GammaVertex(double a, double theta, double k, Random random) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k), random);
    }

    public GammaVertex(DoubleVertex a, DoubleVertex theta, DoubleVertex k) {
        this(a, theta, k, new Random());
    }

    public GammaVertex(DoubleVertex theta, double k) {
        this(new ConstantDoubleVertex(0.0), theta, new ConstantDoubleVertex(k), new Random());
    }

    public GammaVertex(double theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), k, new Random());
    }

    public GammaVertex(double theta, double k) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k), new Random());
    }

    public DoubleVertex getA() {
        return a;
    }

    public DoubleVertex getTheta() {
        return theta;
    }

    public DoubleVertex getK() {
        return k;
    }

    @Override
    public double logPdf(Double value) {
        return Gamma.logPdf(a.getValue(), theta.getValue(), k.getValue(), value);
    }

    @Override
    public Map<String, Double> dLogPdf(Double value) {
        Gamma.Diff diff = Gamma.dlnPdf(a.getValue(), theta.getValue(), k.getValue(), value);
        return convertDualNumbersToDiff(diff.dPda, diff.dPdtheta, diff.dPdk, diff.dPdx);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPda, double dPdtheta, double dPdk, double dPdx) {
        PartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        PartialDerivatives dPdInputsFromTheta = theta.getDualNumber().getPartialDerivatives().multiplyBy(dPdtheta);
        PartialDerivatives dPdInputsFromK = k.getDualNumber().getPartialDerivatives().multiplyBy(dPdk);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromTheta).add(dPdInputsFromK);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public Double sample() {
        return Gamma.sample(a.getValue(), theta.getValue(), k.getValue(), random);
    }

}
