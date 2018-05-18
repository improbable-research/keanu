package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class ExponentialVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex b;

    public ExponentialVertex(DoubleVertex a, DoubleVertex b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    public ExponentialVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    public ExponentialVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b);
    }

    public ExponentialVertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
    }

    public DoubleVertex getA() {
        return a;
    }

    public DoubleVertex getB() {
        return b;
    }

    @Override
    public double logPdf(Double value) {
        return Exponential.logPdf(a.getValue(), b.getValue(), value);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(Double value) {
        Exponential.Diff dP = Exponential.dlnPdf(a.getValue(), b.getValue(), value);
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    @Override
    public Double sample(KeanuRandom random) {
        return Exponential.sample(a.getValue(), b.getValue(), random);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(double dPda, double dPdb, double dPdx) {
        PartialDerivatives dPdInputsFromMu = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        PartialDerivatives dPdInputsFromSigma = b.getDualNumber().getPartialDerivatives().multiplyBy(dPdb);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }

}
