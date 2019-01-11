package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.ChiSquared;
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

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class ChiSquaredVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private IntegerVertex k;
    private static final String K_NAME = "k";
    private static final double LOG_TWO = Math.log(2);

    /**
     * One k that must match a proposed tensor shape of ChiSquared
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param k           the number of degrees of freedom
     */
    public ChiSquaredVertex(@LoadShape long[] tensorShape, @LoadVertexParam(K_NAME) IntegerVertex k) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, k.getShape());

        this.k = k;
        setParents(k);
    }

    public ChiSquaredVertex(long[] tensorShape, int k) {
        this(tensorShape, new ConstantIntegerVertex(k));
    }

    /**
     * One to one constructor for mapping some shape of k to
     * a matching shaped ChiSquared.
     *
     * @param k the number of degrees of freedom
     */
    @ExportVertexToPythonBindings
    public ChiSquaredVertex(IntegerVertex k) {
        this(k.getShape(), k);
    }

    public ChiSquaredVertex(int k) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(k));
    }

    @SaveVertexParam(K_NAME)
    public IntegerVertex getK() {
        return k;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return ChiSquared.withParameters(k.getValue()).sample(shape, random);
    }

    @Override
    public double logProb(DoubleTensor value) {
        return ChiSquared.withParameters(k.getValue()).logProb(value).sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceHolderVertex xPlaceHolder = new LogProbGraph.DoublePlaceHolderVertex(this.getShape());
        final LogProbGraph.IntegerPlaceHolderVertex kPlaceHolder = new LogProbGraph.IntegerPlaceHolderVertex(k.getShape());

        final DoubleVertex halfK = kPlaceHolder.toDouble().div(2.);
        final DoubleVertex numerator = halfK.minus(1.).times(xPlaceHolder.log()).minus(xPlaceHolder.div(2.));
        final DoubleVertex denominator = halfK.times(LOG_TWO).plus(halfK.logGamma());

        final DoubleVertex logProbOutput = numerator.minus(denominator);

        return LogProbGraph.builder()
            .input(this, xPlaceHolder)
            .input(k, kPlaceHolder)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

}
