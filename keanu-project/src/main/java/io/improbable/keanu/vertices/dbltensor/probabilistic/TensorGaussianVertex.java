package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorGaussian;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;
import java.util.Random;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorGaussianVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex sigma;

    /**
     * One mu or sigma or both driving an arbitrarily shaped tensor of Gaussian
     *
     * @param shape  the desired shape of the vertex
     * @param mu     the mu of the Gaussian with either the same shape as specified for this vertex or a scalar
     * @param sigma  the sigma of the Gaussian with either the same shape as specified for this vertex or a scalar
     */
    public TensorGaussianVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex sigma) {

        checkParentShapes(shape, mu.getValue(), sigma.getValue());

        this.mu = mu;
        this.sigma = sigma;
        setParents(mu, sigma);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped gaussian.
     *
     * @param mu     mu with same shape as desired Gaussian tensor or scalar
     * @param sigma  sigma with same shape as desired Gaussian tensor or scalar
     */
    public TensorGaussianVertex(DoubleTensorVertex mu, DoubleTensorVertex sigma) {
        this(getShapeProposal(mu.getValue(), sigma.getValue()), mu, sigma);
    }

    public TensorGaussianVertex(DoubleTensorVertex mu, double sigma) {
        this(mu, new ConstantTensorVertex(sigma));
    }

    public TensorGaussianVertex(double mu, DoubleTensorVertex sigma) {
        this(new ConstantTensorVertex(mu), sigma);
    }

    public TensorGaussianVertex(double mu, double sigma) {
        this(new ConstantTensorVertex(mu), new ConstantTensorVertex(sigma));
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = TensorGaussian.logPdf(muValues, sigmaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorGaussian.Diff dlnP = TensorGaussian.dlnPdf(mu.getValue(), sigma.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPdsigma, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPdsigma,
                                                             DoubleTensor dPdx) {

        TensorPartialDerivatives dPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        TensorPartialDerivatives dPdInputsFromSigma = sigma.getDualNumber().getPartialDerivatives().multiplyBy(dPdsigma);
        TensorPartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorGaussian.sample(getValue().getShape(), mu.getValue(), sigma.getValue(), random);
    }

}
