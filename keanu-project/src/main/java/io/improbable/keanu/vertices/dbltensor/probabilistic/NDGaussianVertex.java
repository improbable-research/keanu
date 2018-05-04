package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.NDGaussian;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class NDGaussianVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex sigma;
    private final KeanuRandom random;

    /**
     * One mu or sigma or both driving an arbitrarily shaped tensor of Gaussian
     *
     * @param shape  the desired shape of the vertex
     * @param mu     the mu of the Gaussian with either the same shape as specified for this vertex or a scalar
     * @param sigma  the sigma of the Gaussian with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public NDGaussianVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex sigma, KeanuRandom random) {

        checkParentShapes(shape, mu.getValue(), sigma.getValue());

        this.mu = mu;
        this.sigma = sigma;
        this.random = random;
        setParents(mu, sigma);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped gaussian.
     *
     * @param mu     mu with same shape as desired Gaussian tensor or scalar
     * @param sigma  sigma with same shape as desired Gaussian tensor or scalar
     * @param random source of randomness
     */
    public NDGaussianVertex(DoubleTensorVertex mu, DoubleTensorVertex sigma, KeanuRandom random) {
        this(getShapeProposal(mu.getValue(), sigma.getValue()), mu, sigma, random);
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = NDGaussian.logPdf(muValues, sigmaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<String, DoubleTensor> dLogPdf(DoubleTensor value) {
        NDGaussian.Diff dlnP = NDGaussian.dlnPdf(mu.getValue(), sigma.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPdsigma, dlnP.dPdx);
    }

    private Map<String, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                               DoubleTensor dPdsigma,
                                                               DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromSigma = sigma.getDualNumber().getPartialDerivatives().multiplyBy(dPdsigma);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if(!this.isObserved()) {
            dPdInputs.getPartialDerivatives().put(getId(), dPdx);
        }

        return dPdInputs.getPartialDerivatives();
    }

    @Override
    public DoubleTensor sample() {
        return NDGaussian.sample(getValue().getShape(), mu.getValue(), sigma.getValue(), random);
    }

}
