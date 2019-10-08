package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.T;
import static io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertexWrapper.wrapIfNeeded;

public class StudentTVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final IntegerVertex v;
    private static final String V_NAME = "v";

    /**
     * One v that must match a proposed tensor shape of StudentT
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape expected tensor shape
     * @param v           Degrees of Freedom
     */
    public StudentTVertex(@LoadShape long[] tensorShape,
                          @LoadVertexParam(V_NAME) Vertex<IntegerTensor, ?> v) {
        super(TensorShape.getBroadcastResultShape(tensorShape, v.getShape()));
        this.v = wrapIfNeeded(v);
        setParents(v);
    }

    public StudentTVertex(long[] tensorShape, int v) {
        this(tensorShape, new ConstantIntegerVertex(v));
    }

    @ExportVertexToPythonBindings
    public StudentTVertex(IntegerVertex v) {
        this(v.getShape(), v);
    }

    public StudentTVertex(int v) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(v));
    }

    @SaveVertexParam(V_NAME)
    public IntegerVertex getV() {
        return v;
    }

    @Override
    public double logProb(DoubleTensor t) {
        return StudentT.withParameters(v.getValue()).logProb(t).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final IntegerPlaceholderVertex vPlaceholder = new IntegerPlaceholderVertex(v.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(v, vPlaceholder)
            .logProbOutput(StudentT.logProbOutput(xPlaceholder, vPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor t, Set<? extends Vertex> withRespect) {
        Map<Vertex, DoubleTensor> m = new HashMap<>();

        if (withRespect.contains(this)) {
            Diffs diff = StudentT.withParameters(v.getValue()).dLogProb(t);
            m.put(this, diff.get(T).getValue());
        }

        return m;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return StudentT.withParameters(v.getValue()).sample(shape, random);
    }
}
