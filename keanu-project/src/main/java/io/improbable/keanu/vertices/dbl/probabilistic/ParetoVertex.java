package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Pareto;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.L;
import static io.improbable.keanu.distributions.hyperparam.Diffs.S;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class ParetoVertex extends DoubleVertex implements SaveableVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor> {

    private final DoubleVertex scale;
    private final DoubleVertex location;
    private static final String SCALE_NAME = "scale";
    private static final String LOCATION_NAME = "location";

    /**
     * Provides a Vertex implementing the Pareto Distribution.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param location    the location value(s) of the Pareto.  Must either be the same shape as tensorShape or a scalar
     * @param scale       the scale value(s) of the Pareto.  Must either be the same shape as tensorShape or a scalar
     */
    public ParetoVertex(long[] tensorShape, DoubleVertex location, DoubleVertex scale) {
        super(tensorShape);
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), scale.getShape());

        this.scale = scale;
        this.location = location;
        setParents(location, scale);
    }

    @ExportVertexToPythonBindings
    public ParetoVertex(@LoadParentVertex(LOCATION_NAME) DoubleVertex location,
                        @LoadParentVertex(SCALE_NAME) DoubleVertex scale) {
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

    public ParetoVertex(long[] tensorShape, double location, DoubleVertex scale) {
        this(tensorShape, new ConstantDoubleVertex(location), scale);
    }

    public ParetoVertex(long[] tensorShape, DoubleVertex location, double scale) {
        this(tensorShape, location, new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(long[] tensorShape, double location, double scale) {
        this(tensorShape, new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    @SaveParentVertex(SCALE_NAME)
    public DoubleVertex getScale() {
        return scale;
    }

    @SaveParentVertex(LOCATION_NAME)
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
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Pareto.withParameters(location.getValue(), scale.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(location)) {
            dLogProbWrtParameters.put(location, dlnP.get(L).getValue());
        }

        if (withRespectTo.contains(scale)) {
            dLogProbWrtParameters.put(scale, dlnP.get(S).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Pareto.withParameters(location.getValue(), scale.getValue()).sample(shape, random);
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        if (isObserved()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            return PartialDerivatives.withRespectToSelf(this.getId(), this.getShape());
        }
    }
}
