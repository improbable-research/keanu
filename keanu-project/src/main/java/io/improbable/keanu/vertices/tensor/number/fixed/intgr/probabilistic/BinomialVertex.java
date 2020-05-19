package io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
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
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertexWrapper;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

public class BinomialVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor>, LogProbGraphSupplier {

    private final DoubleVertex p;
    private final IntegerVertex n;
    private final static String P_NAME = "p";
    private final static String N_NAME = "n";

    public BinomialVertex(@LoadShape long[] tensorShape,
                          @LoadVertexParam(P_NAME) Vertex<DoubleTensor, ?> p,
                          @LoadVertexParam(N_NAME) Vertex<IntegerTensor, ?> n) {
        super(TensorShape.getBroadcastResultShape(tensorShape, p.getShape(), n.getShape()));
        this.p = DoubleVertexWrapper.wrapIfNeeded(p);
        this.n = IntegerVertexWrapper.wrapIfNeeded(n);

        setParents(p, n);
    }

    public BinomialVertex(long[] tensorShape, double p, Vertex<IntegerTensor, ?> n) {
        this(tensorShape, ConstantVertex.of(p), n);
    }

    public BinomialVertex(long[] tensorShape, Vertex<DoubleTensor, ?> p, int n) {
        this(tensorShape, p, ConstantVertex.of(n));
    }

    public BinomialVertex(long[] tensorShape, double p, int n) {
        this(tensorShape, ConstantVertex.of(p), ConstantVertex.of(n));
    }

    @ExportVertexToPythonBindings
    public BinomialVertex(Vertex<DoubleTensor, ?> p, Vertex<IntegerTensor, ?> n) {
        this(TensorShape.getBroadcastResultShape(p.getShape(), n.getShape()), p, n);
    }

    public BinomialVertex(double p, Vertex<IntegerTensor, ?> n) {
        this(ConstantVertex.of(p), n);
    }

    public BinomialVertex(Vertex<DoubleTensor, ?> p, int n) {
        this(p, ConstantVertex.of(n));
    }

    public BinomialVertex(double p, int n) {
        this(ConstantVertex.of(p), ConstantVertex.of(n));
    }

    @Override
    public double logProb(IntegerTensor k) {
        return Binomial.withParameters(p.getValue(), n.getValue()).logProb(k).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceholderVertex kPlaceholder = new IntegerPlaceholderVertex(this.getShape());
        DoublePlaceholderVertex pPlaceholder = new DoublePlaceholderVertex(p.getShape());
        IntegerPlaceholderVertex nPlaceholder = new IntegerPlaceholderVertex(n.getShape());

        return LogProbGraph.builder()
            .input(this, kPlaceholder)
            .input(p, pPlaceholder)
            .input(n, nPlaceholder)
            .logProbOutput(Binomial.logProbOutput(kPlaceholder, pPlaceholder, nPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        boolean wrtP = withRespectTo.contains(p);

        DoubleTensor[] result = Binomial.withParameters(p.getValue(), n.getValue()).dLogProb(value, wrtP);
        if (wrtP) {
            return Collections.singletonMap(p, result[0]);
        } else {
            return Collections.emptyMap();
        }
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Binomial.withParameters(p.getValue(), n.getValue()).sample(shape, random);
    }

    @SaveVertexParam(P_NAME)
    public DoubleVertex getP() {
        return p;
    }

    @SaveVertexParam(N_NAME)
    public IntegerVertex getN() {
        return n;
    }
}
