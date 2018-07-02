package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class GaussianVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;

    /**
     * One mu or sigma or both driving an arbitrarily shaped tensor of Gaussian
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param mu          the mu of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     * @param sigma       the sigma of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     */
    public GaussianVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex sigma) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), sigma.getShape());

        this.mu = mu;
        this.sigma = sigma;
        setParents(mu, sigma);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public GaussianVertex(DoubleVertex mu, DoubleVertex sigma) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public GaussianVertex(DoubleVertex mu, double sigma) {
        this(mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(double mu, DoubleVertex sigma) {
        this(new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(double mu, double sigma) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(int[] tensorShape, DoubleVertex mu, double sigma) {
        this(tensorShape, mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(int[] tensorShape, double mu, DoubleVertex sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(int[] tensorShape, double mu, double sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public DoubleVertex getMu() {
        return mu;
    }

    public DoubleVertex getSigma() {
        return sigma;
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = Gaussian.logPdf(muValues, sigmaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        Gaussian.Diff dlnP = Gaussian.dlnPdf(mu.getValue(), sigma.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPdsigma, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPdsigma,
                                                             DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromSigma = sigma.getDualNumber().getPartialDerivatives().multiplyBy(dPdsigma);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gaussian.sample(getShape(), mu.getValue(), sigma.getValue(), random);
    }

}
