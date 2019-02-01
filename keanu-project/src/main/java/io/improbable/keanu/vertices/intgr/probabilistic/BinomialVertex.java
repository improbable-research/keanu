package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class BinomialVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private final DoubleVertex p;
    private final IntegerVertex n;
    private final static String P_NAME = "p";
    private final static String N_NAME = "n";

    public BinomialVertex(@LoadShape long[] tensorShape,
                          @LoadVertexParam(P_NAME) DoubleVertex p,
                          @LoadVertexParam(N_NAME) IntegerVertex n) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, p.getShape(), n.getShape());
        this.p = p;
        this.n = n;

        setParents(p, n);
    }

    public BinomialVertex(long[] tensorShape, double p, IntegerVertex n) {
        this(tensorShape, ConstantVertex.of(p), n);
    }

    public BinomialVertex(long[] tensorShape, DoubleVertex p, int n) {
        this(tensorShape, p, ConstantVertex.of(n));
    }

    public BinomialVertex(long[] tensorShape, double p, int n) {
        this(tensorShape, ConstantVertex.of(p), ConstantVertex.of(n));
    }

    @ExportVertexToPythonBindings
    public BinomialVertex(DoubleVertex p, IntegerVertex n) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(p.getShape(), n.getShape()), p, n);
    }

    public BinomialVertex(double p, IntegerVertex n) {
        this(ConstantVertex.of(p), n);
    }

    public BinomialVertex(DoubleVertex p, int n) {
        this(p, ConstantVertex.of(n));
    }

    public BinomialVertex(double p, int n) {
        this(ConstantVertex.of(p), ConstantVertex.of(n));
    }

    @Override
    public double logProb(IntegerTensor kTensor) {
        return Binomial.withParameters(p.getValue(), n.getValue()).logProb(kTensor).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
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
