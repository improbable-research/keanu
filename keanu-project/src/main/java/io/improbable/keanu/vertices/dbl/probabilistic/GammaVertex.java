package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class GammaVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex theta;
    private final DoubleVertex k;

    /**
     * @param a     location
     * @param theta scale
     * @param k     shape
     */
    public GammaVertex(DoubleVertex a, DoubleVertex theta, DoubleVertex k) {
        this.a = a;
        this.theta = theta;
        this.k = k;
        setParents(a, theta, k);
    }

    public GammaVertex(double a, double theta, double k) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    public GammaVertex(DoubleVertex theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(0.0), theta, k);
    }

    public GammaVertex(DoubleVertex theta, double k) {
        this(new ConstantDoubleVertex(0.0), theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(double theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(double theta, double k) {
        this(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
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
    public Map<Long, DoubleTensor> dLogPdf(Double value) {
        Gamma.Diff diff = Gamma.dlnPdf(a.getValue(), theta.getValue(), k.getValue(), value);
        return convertDualNumbersToDiff(diff.dPda, diff.dPdtheta, diff.dPdk, diff.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(double dPda, double dPdtheta, double dPdk, double dPdx) {
        PartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        PartialDerivatives dPdInputsFromTheta = theta.getDualNumber().getPartialDerivatives().multiplyBy(dPdtheta);
        PartialDerivatives dPdInputsFromK = k.getDualNumber().getPartialDerivatives().multiplyBy(dPdk);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromTheta).add(dPdInputsFromK);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }

    @Override
    public Double sample(KeanuRandom random) {
        return Gamma.sample(a.getValue(), theta.getValue(), k.getValue(), random);
    }

}
