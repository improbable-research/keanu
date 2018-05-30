package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class LaplaceVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex beta;

    public LaplaceVertex(DoubleVertex mu, DoubleVertex beta) {
        this.mu = mu;
        this.beta = beta;
        setParents(mu, beta);
    }

    public LaplaceVertex(DoubleVertex mu, double beta) {
        this(mu, new ConstantDoubleVertex(beta));
    }

    public LaplaceVertex(double mu, DoubleVertex beta) {
        this(new ConstantDoubleVertex(mu), beta);
    }

    public LaplaceVertex(double mu, double beta) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
    }

    @Override
    public double logPdf(Double value) {
        return Laplace.logPdf(mu.getValue(), beta.getValue(), value);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(Double value) {
        Laplace.Diff diff = Laplace.dlnPdf(mu.getValue(), beta.getValue(), value);
        return convertDualNumbersToDiff(diff.dPdmu, diff.dPdbeta, diff.dPdx);
    }

    @Override
    public Double sample(KeanuRandom random) {
        return Laplace.sample(mu.getValue(), beta.getValue(), random);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(double dPdmu, double dPdbeta, double dPdx) {
        PartialDerivatives dPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromSigma = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }
}
