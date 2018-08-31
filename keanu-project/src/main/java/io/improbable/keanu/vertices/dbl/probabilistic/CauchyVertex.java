package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Cauchy;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class CauchyVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex location;
    private final DoubleVertex scale;

    /**
     * One location or scale or both that match a proposed tensor shape of Cauchy
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param location    the location of the Cauchy with either the same tensorShape as specified for this vertex or a scalar
     * @param scale       the scale of the Cauchy with either the same tensorShape as specified for this vertex or a scalar
     */
    public CauchyVertex(int[] tensorShape, DoubleVertex location, DoubleVertex scale) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), scale.getShape());

        this.location = location;
        this.scale = scale;
        setParents(location, scale);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public CauchyVertex(DoubleVertex location, DoubleVertex scale) {
        this(checkHasSingleNonScalarShapeOrAllScalar(location.getShape(), scale.getShape()), location, scale);
    }

    public CauchyVertex(DoubleVertex location, double scale) {
        this(location, new ConstantDoubleVertex(scale));
    }

    public CauchyVertex(double location, DoubleVertex scale) {
        this(new ConstantDoubleVertex(location), scale);
    }

    public CauchyVertex(double location, double scale) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    public CauchyVertex(int[] tensorShape, DoubleVertex location, double scale) {
        this(tensorShape, location, new ConstantDoubleVertex(scale));
    }

    public CauchyVertex(int[] tensorShape, double location, DoubleVertex scale) {
        this(tensorShape, new ConstantDoubleVertex(location), scale);
    }

    public CauchyVertex(int[] tensorShape, double location, double scale) {
        this(tensorShape, new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    public DoubleVertex getLocation() {
        return location;
    }

    public DoubleVertex getScale() {
        return scale;
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor locationValues = location.getValue();
        DoubleTensor scaleValues = scale.getValue();

        DoubleTensor logPdfs = Cauchy.withParameters(locationValues, scaleValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<VertexId, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Cauchy.withParameters(location.getValue(), scale.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(L).getValue(), dlnP.get(S).getValue(), dlnP.get(X).getValue());
    }

    private Map<VertexId, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdlocation,
                                                             DoubleTensor dLogPdscale,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromLocation = location.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdlocation);
        PartialDerivatives dLogPdInputsFromScale = scale.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdscale);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromLocation.add(dLogPdInputsFromScale);

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
        return Cauchy.withParameters(location.getValue(), scale.getValue()).sample(getShape(), random);
    }
}
