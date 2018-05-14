package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.InverseGamma;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Map;
import java.util.Random;

public class InverseGammaVertex extends ProbabilisticDouble {

    private DoubleVertex a;
    private DoubleVertex b;

    public InverseGammaVertex(DoubleVertex a, DoubleVertex b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    public InverseGammaVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    public InverseGammaVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b);
    }

    public InverseGammaVertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
    }

    @Override
    public Double sample(Random random) {
        return InverseGamma.sample(a.getValue(), b.getValue(), random);
    }

    @Override
    public double logPdf(Double value) {
        return InverseGamma.logPdf(a.getValue(), b.getValue(), value);
    }

    public Map<String, DoubleTensor> dLogPdf(Double value) {
        InverseGamma.Diff dP = InverseGamma.dlnPdf(a.getValue(), b.getValue(), value);
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    private Map<String, DoubleTensor> convertDualNumbersToDiff(double dPda, double dPdb, double dPdx) {
        PartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        PartialDerivatives dPdInputsFromB = b.getDualNumber().getPartialDerivatives().multiplyBy(dPdb);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }

}
