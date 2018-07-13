package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;


import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

public class LogisticVertex extends DoubleVertex implements ProbabilisticDouble {

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
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(LogisticVertex.class));

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
    public double logProb(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor sValues = s.getValue();

        DoubleTensor logPdfs = Logistic.withParameters(muValues, sValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Logistic.withParameters(mu.getValue(), s.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(MU).getValue(), dlnP.get(S).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdmu,
                                                             DoubleTensor dLogPds,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdmu);
        PartialDerivatives dLogPdInputsFromS = s.getDualNumber().getPartialDerivatives().multiplyBy(dLogPds);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromMu.add(dLogPdInputsFromS);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Logistic.withParameters(mu.getValue(), s.getValue()).sample(getShape(), random);
    }
}
