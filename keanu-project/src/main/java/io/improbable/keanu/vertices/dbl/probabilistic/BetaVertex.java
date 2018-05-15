package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Map;
import java.util.Random;

public class BetaVertex extends ProbabilisticDouble {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;

    public BetaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
    }

    public BetaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    @Override
    public double logPdf(Double value) {
        return Beta.logPdf(alpha.getValue(), beta.getValue(), value);
    }

    public DoubleVertex getAlpha() {
        return alpha;
    }

    public DoubleVertex getBeta() {
        return beta;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(Double value) {
        Beta.Diff dlnPdf = Beta.dlnPdf(alpha.getValue(), beta.getValue(), value);
        return convertDualNumbersToDiff(dlnPdf.dPdAlpha, dlnPdf.dPdBeta, dlnPdf.dPdx);
    }

    public Map<Long, DoubleTensor> convertDualNumbersToDiff(double dPdAlpha, double dPdBeta, double dPdx) {
        PartialDerivatives dPdInputsFromAlpha = alpha.getDualNumber().getPartialDerivatives().multiplyBy(dPdAlpha);
        PartialDerivatives dPdInputsFromBeta = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdBeta);
        PartialDerivatives dPdInputs = dPdInputsFromAlpha.add(dPdInputsFromBeta);

        if (!isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }

    @Override
    public Double sample(Random random) {
        return Beta.sample(alpha.getValue(), beta.getValue(), 0, 1, random);
    }

}
