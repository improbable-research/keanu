package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Pareto;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ParetoVertex extends DoubleVertex implements ProbabilisticDouble {
    private final DoubleVertex alpha;
    private final DoubleVertex xm;

    public ParetoVertex(int[] tensorShape, DoubleVertex xm, DoubleVertex alpha) {
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, xm.getShape(), alpha.getShape());

        this.alpha = alpha;
        this.xm = xm;
        setParents(xm, alpha);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public ParetoVertex(DoubleVertex xm, DoubleVertex alpha) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xm.getShape(), alpha.getShape()), xm, alpha);
    }

    public ParetoVertex(double xm, DoubleVertex alpha) {
        this(new ConstantDoubleVertex(xm), alpha);
    }

    public ParetoVertex(DoubleVertex xm, double alpha) {
        this(xm, new ConstantDoubleVertex(alpha));
    }

    public ParetoVertex(double xm, double alpha) {
        this(new ConstantDoubleVertex(xm), new ConstantDoubleVertex(alpha));
    }

    public ParetoVertex(int[] tensorShape, double xm, DoubleVertex alpha) {
        this(tensorShape, new ConstantDoubleVertex(xm), alpha);
    }

    public ParetoVertex(int[] tensorShape, DoubleVertex xm, double alpha) {
        this(tensorShape, xm, new ConstantDoubleVertex(alpha));
    }

    public ParetoVertex(int[] tensorShape, double xm, double alpha) {
        this(tensorShape, new ConstantDoubleVertex(xm), new ConstantDoubleVertex(alpha));
    }

    public DoubleVertex getAlpha() {
        return alpha;
    }

    public DoubleVertex getXm() {
        return xm;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor xmValues = xm.getValue();
        DoubleTensor alphaValues = alpha.getValue();

        DoubleTensor logPdfs = Pareto.withParameters(xmValues, alphaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Pareto.withParameters(xm.getValue(), alpha.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(L).getValue(), dlnP.get(S).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdXm,
                                                             DoubleTensor dLogPdAlpha,
                                                             DoubleTensor dLogPdX) {

        PartialDerivatives dLogPdInputsFromXm = xm.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdXm);
        PartialDerivatives dLogPdInputsFromAlpha = alpha.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdAlpha);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromXm.add(dLogPdInputsFromAlpha);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdX.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdX.getShape(), dLogPdX.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));

        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Pareto.withParameters(xm.getValue(), alpha.getValue()).sample(getShape(), random);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }
}
