package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.T;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static java.lang.Math.PI;

public class StudentTVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private static final double HALF_LOG_PI = java.lang.Math.log(PI) / 2;
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
    public StudentTVertex(@LoadShape long[] tensorShape, @LoadVertexParam(V_NAME) IntegerVertex v) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, v.getShape());
        this.v = v;
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
        return StudentT.withParameters(v.getValue()).logProb(t).sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceholderVertex xPlaceholder = new LogProbGraph.DoublePlaceholderVertex(this.getShape());
        final LogProbGraph.IntegerPlaceHolderVertex tPlaceholder = new LogProbGraph.IntegerPlaceHolderVertex(this.getShape());

        DoubleVertex vAsDouble = tPlaceholder.toDouble();
        DoubleVertex halfVPlusOne = vAsDouble.plus(1).div(2);

        DoubleVertex logGammaHalfVPlusOne = halfVPlusOne.logGamma();
        DoubleVertex logGammaHalfV = vAsDouble.div(2).logGamma();
        DoubleVertex halfLogV = vAsDouble.log().div(2);

        DoubleVertex logProbOutput = logGammaHalfVPlusOne
            .minus(halfLogV)
            .minus(HALF_LOG_PI)
            .minus(logGammaHalfV)
            .minus(
                halfVPlusOne.times(
                    xPlaceholder.pow(2).div(vAsDouble).plus(1).log()
                )
            );

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(v, tPlaceholder)
            .logProbOutput(logProbOutput)
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
