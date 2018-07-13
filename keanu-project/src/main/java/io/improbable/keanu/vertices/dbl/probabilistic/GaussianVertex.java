package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class GaussianVertex extends ProbabilisticDouble implements Differentiable, Probabilistic<DoubleTensor> {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;

    /**
     * One mu or sigma or both that match a proposed tensor shape of Gaussian
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
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
    public double logProb(DoubleTensor value) {
        return Gaussian.withParameters(mu.getValue(), sigma.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        List<DoubleTensor> dlnP = Gaussian.withParameters(mu.getValue(), sigma.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(0), dlnP.get(1), dlnP.get(2));
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPdsigma,
                                                             DoubleTensor dPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dPdInputsFromMu = differentiator.calculateDual((Differentiable) mu).getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromSigma = differentiator.calculateDual((Differentiable) sigma).getPartialDerivatives().multiplyBy(dPdsigma);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }


    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gaussian.withParameters(mu.getValue(), sigma.getValue()).sample(getShape(), random);
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }
}
