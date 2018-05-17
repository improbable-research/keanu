package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorLaplace;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorLaplaceVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex beta;
    private final KeanuRandom random;

    /**
     * One mu or beta or both driving an arbitrarily shaped tensor of Laplace
     *
     * @param shape  the desired shape of the vertex
     * @param mu     the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta   the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorLaplaceVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex beta, KeanuRandom random) {

        checkParentShapes(shape, mu.getValue(), beta.getValue());

        this.mu = mu;
        this.beta = beta;
        this.random = random;
        setParents(mu, beta);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped laplace.
     *
     * @param mu     the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta   the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorLaplaceVertex(DoubleTensorVertex mu, DoubleTensorVertex beta, KeanuRandom random) {
        this(getShapeProposal(mu.getValue(), beta.getValue()), mu, beta, random);
    }

    public TensorLaplaceVertex(DoubleTensorVertex mu, double beta, KeanuRandom random) {
        this(mu, new ConstantTensorVertex(beta), random);
    }

    public TensorLaplaceVertex(double mu, DoubleTensorVertex beta, KeanuRandom random) {
        this(new ConstantTensorVertex(mu), beta, random);
    }

    public TensorLaplaceVertex(double mu, double beta, KeanuRandom random) {
        this(new ConstantTensorVertex(mu), new ConstantTensorVertex(beta), random);
    }

    public TensorLaplaceVertex(DoubleTensorVertex mu, DoubleTensorVertex beta) {
        this(getShapeProposal(mu.getValue(), beta.getValue()), mu, beta, new KeanuRandom());
    }

    public TensorLaplaceVertex(DoubleTensorVertex mu, double beta) {
        this(mu, new ConstantTensorVertex(beta), new KeanuRandom());
    }

    public TensorLaplaceVertex(double mu, DoubleTensorVertex beta) {
        this(new ConstantTensorVertex(mu), beta, new KeanuRandom());
    }

    public TensorLaplaceVertex(double mu, double beta) {
        this(new ConstantTensorVertex(mu), new ConstantTensorVertex(beta), new KeanuRandom());
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = TensorLaplace.logPdf(muValues, betaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorLaplace.Diff dlnP = TensorLaplace.dlnPdf(mu.getValue(), beta.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPdbeta, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPdbeta,
                                                             DoubleTensor dPdx) {

        TensorPartialDerivatives dPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        TensorPartialDerivatives dPdInputsFromSigma = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdbeta);
        TensorPartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample() {
        return TensorLaplace.sample(getValue().getShape(), mu.getValue(), beta.getValue(), random);
    }
}
