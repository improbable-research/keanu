package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.Tensor.SCALAR_SHAPE;
import static io.improbable.keanu.tensor.TensorShape.concat;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class LogisticVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex s;

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of Logistic
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu          the mu (location) of the Logistic with either the same shape as specified for this vertex or mu scalar
     * @param s           the s (scale) of the Logistic with either the same shape as specified for this vertex or mu scalar
     */
    public LogisticVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex s) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), s.getShape());

        this.mu = mu;
        this.s = s;
        setParents(mu, s);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public LogisticVertex(DoubleVertex mu, DoubleVertex s) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), s.getShape()), mu, s);
    }

    public LogisticVertex(DoubleVertex mu, double s) {
        this(mu, new ConstantDoubleVertex(s));
    }

    public LogisticVertex(double mu, DoubleVertex s) {
        this(new ConstantDoubleVertex(mu), s);
    }

    public LogisticVertex(double mu, double s) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(s));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor sValues = s.getValue();

        DoubleTensor logPdfs = Logistic.logPdf(muValues, sValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        Logistic.DiffLogP dlnP = Logistic.dlnPdf(mu.getValue(), s.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dLogPdmu, dlnP.dLogPds, dlnP.dLogPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdmu,
                                                             DoubleTensor dLogPds,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdmu);
        PartialDerivatives dLogPdInputsFromS = s.getDualNumber().getPartialDerivatives().multiplyBy(dLogPds);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromMu.add(dLogPdInputsFromS);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(concat(SCALAR_SHAPE, dLogPdx.getShape())));
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Logistic.sample(getShape(), mu.getValue(), s.getValue(), random);
    }
}
