package io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Geometric;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class GeometricVertex extends VertexImpl<IntegerTensor, IntegerVertex>
    implements ProbabilisticInteger, LogProbGraphSupplier {

    private final DoubleVertex p;
    private final static String P_NAME = "p";

    /**
     * A Vertex representing a Geometrically distributed random variable.
     * <p>
     * The Keanu Implementation has a support of {1, 2, 3, ...} i.e. it produces the number of tests until success (not
     * the number of failures until success which has a support {0, 1, 2, ...}
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param p           the probability that an individual test is a success
     */
    public GeometricVertex(@LoadShape long[] tensorShape,
                           @LoadVertexParam(P_NAME) Vertex<DoubleTensor, ?> p) {
        super(TensorShape.getBroadcastResultShape(tensorShape, p.getShape()));
        this.p = wrapIfNeeded(p);

        setParents(p);
    }

    public GeometricVertex(long[] tensorShape, double p) {
        this(tensorShape, ConstantVertex.of(p));
    }

    @ExportVertexToPythonBindings
    public GeometricVertex(Vertex<DoubleTensor, ?> p) {
        this(p.getShape(), p);
    }

    public GeometricVertex(double p) {
        this(ConstantVertex.of(p));
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {
        return Geometric.withParameters(p.getValue()).sample(shape, random);
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Geometric.withParameters(p.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceholderVertex valuePlaceholder = new IntegerPlaceholderVertex(this.getShape());
        DoublePlaceholderVertex pPlaceholder = new DoublePlaceholderVertex(p.getShape());

        return LogProbGraph.builder()
            .input(this, valuePlaceholder)
            .input(p, pPlaceholder)
            .output(Geometric.logProbOutput(valuePlaceholder, pPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        boolean wrtP = withRespectTo.contains(p);

        DoubleTensor[] result = Geometric.withParameters(p.getValue()).dLogProb(value, wrtP);
        if (wrtP) {
            return Collections.singletonMap(p, result[0]);
        } else {
            return Collections.emptyMap();
        }
    }

    @SaveVertexParam(P_NAME)
    public DoubleVertex getP() {
        return p;
    }
}
