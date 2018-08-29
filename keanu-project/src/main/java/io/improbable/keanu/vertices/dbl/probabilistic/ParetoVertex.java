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

    private final DoubleVertex scale;
    private final DoubleVertex location;

    /**
     * Provides a Vertex implementing the Pareto Distribution.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param location    the location value(s) of the Pareto.  Must either be the same shape as tensorShape or a scalar
     * @param scale       the scale value(s) of the Pareto.  Must either be the same shape as tensorShape or a scalar
     */
    public ParetoVertex(int[] tensorShape, DoubleVertex location, DoubleVertex scale) {
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), scale.getShape());

        this.scale = scale;
        this.location = location;
        setParents(location, scale);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public ParetoVertex(DoubleVertex location, DoubleVertex scale) {
        this(checkHasSingleNonScalarShapeOrAllScalar(location.getShape(), scale.getShape()), location, scale);
    }

    public ParetoVertex(double location, DoubleVertex scale) {
        this(new ConstantDoubleVertex(location), scale);
    }

    public ParetoVertex(DoubleVertex location, double scale) {
        this(location, new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(double location, double scale) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(int[] tensorShape, double location, DoubleVertex scale) {
        this(tensorShape, new ConstantDoubleVertex(location), scale);
    }

    public ParetoVertex(int[] tensorShape, DoubleVertex location, double scale) {
        this(tensorShape, location, new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(int[] tensorShape, double location, double scale) {
        this(tensorShape, new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    public DoubleVertex getScale() {
        return scale;
    }

    public DoubleVertex getLocation() {
        return location;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor locValues = location.getValue();
        DoubleTensor scaleValues = scale.getValue();

        DoubleTensor logPdfs = Pareto.withParameters(locValues, scaleValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Pareto.withParameters(location.getValue(), scale.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(L).getValue(), dlnP.get(S).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdLocation,
                                                             DoubleTensor dLogPdScale,
                                                             DoubleTensor dLogPdX) {

        PartialDerivatives dLogPdInputsFromLocation = location.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdLocation);
        PartialDerivatives dLogPdInputsFromScale = scale.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdScale);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromLocation.add(dLogPdInputsFromScale);

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
        return Pareto.withParameters(location.getValue(), scale.getValue()).sample(getShape(), random);
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
