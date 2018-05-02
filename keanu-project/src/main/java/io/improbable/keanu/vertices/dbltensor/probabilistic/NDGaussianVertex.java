package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.NDGaussian;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

public class NDGaussianVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex sigma;
    private final KeanuRandom random;

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped gaussian.
     *
     * @param mu     mu with same shape as desired Gaussian tensor
     * @param sigma  sigma with same shape as desired Gasussian tensor
     * @param random source of randomness
     */
    public NDGaussianVertex(DoubleTensorVertex mu, DoubleTensorVertex sigma, KeanuRandom random) {
        if (mu.getValue().hasSameShapeAs(sigma.getValue())) {
            throw new IllegalArgumentException("mu and sigma must match shape");
        }

        this.mu = mu;
        this.sigma = sigma;
        this.random = random;
        setParents(mu, sigma);
    }

    //One mu or sigma or both driving an arbitrarily shaped tensor of gaussians
//    public GaussianVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex sigma, Random random) {
//
//    }


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

        dPdInputs.getPartialDerivatives().put(getId(), dPdx);

        return dPdInputs.getPartialDerivatives();
    }

    @Override
    public DoubleTensor sample() {
        return NDGaussian.sample(mu.getValue(), sigma.getValue(), random);
    }

}
