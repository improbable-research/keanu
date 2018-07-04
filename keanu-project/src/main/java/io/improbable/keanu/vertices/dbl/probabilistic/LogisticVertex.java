package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class LogisticVertex extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex s;

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of Logistic
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
        Logistic.Diff dlnP = Logistic.dlnPdf(mu.getValue(), s.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dPdmu, dlnP.dPds, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPds,
                                                             DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromA = mu.getDualNumber().getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromB = s.getDualNumber().getPartialDerivatives().multiplyBy(dPds);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Logistic.sample(getShape(), mu.getValue(), s.getValue(), random);
    }
}
