package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorLogNormal;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class LogNormalVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;

    public LogNormalVertex(int[] shape, DoubleVertex mu, DoubleVertex sigma) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape(), sigma.getShape());

        this.mu = mu;
        this.sigma = sigma;
        setParents(mu, sigma);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public LogNormalVertex(int[] shape, DoubleVertex mu, double sigma) {
        this(shape, mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(int[] shape, double mu, DoubleVertex sigma) {
        this(shape, ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(int[] shape, double mu, double sigma) {
        this(shape, ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    public LogNormalVertex(DoubleVertex mu, DoubleVertex sigma) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public LogNormalVertex(double mu, DoubleVertex sigma) {
        this(ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(DoubleVertex mu, double sigma) {
        this(mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(double mu, double sigma) {
        this(ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = TensorLogNormal.logPdf(muValues, sigmaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorLogNormal.Diff dlnP = TensorLogNormal.dlnPdf(mu.getValue(), sigma.getValue(), value);
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
        return TensorLogNormal.sample(getShape(), mu.getValue(), sigma.getValue(), random);
    }
}
