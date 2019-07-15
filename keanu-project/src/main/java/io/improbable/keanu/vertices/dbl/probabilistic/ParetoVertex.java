package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Pareto;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.L;
import static io.improbable.keanu.distributions.hyperparam.Diffs.S;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class ParetoVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

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
    public ParetoVertex(@LoadShape long[] tensorShape,
                        @LoadVertexParam(LOCATION_NAME) Vertex<DoubleTensor, ?> location,
                        @LoadVertexParam(SCALE_NAME) Vertex<DoubleTensor, ?> scale) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, location.getShape(), scale.getShape());

        this.scale = wrapIfNeeded(scale);
        this.location = wrapIfNeeded(location);
        setParents(location, scale);
    }

    @ExportVertexToPythonBindings
    public ParetoVertex(Vertex<DoubleTensor, ?> location, Vertex<DoubleTensor, ?> scale) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(location.getShape(), scale.getShape()), location, scale);
    }

    public ParetoVertex(double location, Vertex<DoubleTensor, ?> scale) {
        this(new ConstantDoubleVertex(location), scale);
    }

    public ParetoVertex(Vertex<DoubleTensor, ?> location, double scale) {
        this(location, new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(double location, double scale) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(long[] tensorShape, double location, Vertex<DoubleTensor, ?> scale) {
        this(tensorShape, new ConstantDoubleVertex(location), scale);
    }

    public ParetoVertex(long[] tensorShape, Vertex<DoubleTensor, ?> location, double scale) {
        this(tensorShape, location, new ConstantDoubleVertex(scale));
    }

    public ParetoVertex(long[] tensorShape, double location, double scale) {
        this(tensorShape, new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
    }

    @SaveVertexParam(SCALE_NAME)
    public DoubleVertex getScale() {
        return scale;
    }

    @SaveVertexParam(LOCATION_NAME)
    public DoubleVertex getLocation() {
        return location;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor locValues = location.getValue();
        DoubleTensor scaleValues = scale.getValue();

        DoubleTensor logPdfs = Pareto.withParameters(locValues, scaleValues).logProb(value);

        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex locationPlaceholder = new DoublePlaceholderVertex(location.getShape());
        final DoublePlaceholderVertex scalePlaceholder = new DoublePlaceholderVertex(scale.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(location, locationPlaceholder)
            .input(scale, scalePlaceholder)
            .logProbOutput(Pareto.logProbOutput(xPlaceholder, locationPlaceholder, scalePlaceholder))
            .build();
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

}
