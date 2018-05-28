package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorLaplace;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class TensorLaplaceVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex beta;

    /**
     * One mu or beta or both driving an arbitrarily shaped tensor of Laplace
     *
     * @param shape  the desired shape of the vertex
     * @param mu     the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta   the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    public TensorLaplaceVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex beta) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape(), beta.getShape());

        this.mu = mu;
        this.beta = beta;
        setParents(mu, beta);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped laplace.
     *
     * @param mu     the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta   the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    public TensorLaplaceVertex(DoubleTensorVertex mu, DoubleTensorVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), beta.getShape()), mu, beta);
    }

    public TensorLaplaceVertex(DoubleTensorVertex mu, double beta) {
        this(mu, new ConstantTensorVertex(beta));
    }

    public TensorLaplaceVertex(double mu, DoubleTensorVertex beta) {
        this(new ConstantTensorVertex(mu), beta);
    }

    public TensorLaplaceVertex(double mu, double beta) {
        this(new ConstantTensorVertex(mu), new ConstantTensorVertex(beta));
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
        TensorPartialDerivatives dPdInputsFromBeta = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdbeta);
        TensorPartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromBeta);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorLaplace.sample(getShape(), mu.getValue(), beta.getValue(), random);
    }

}
