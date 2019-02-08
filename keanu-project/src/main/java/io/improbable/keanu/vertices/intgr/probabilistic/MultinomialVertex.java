package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraph.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.LogProbGraph.IntegerPlaceHolderVertex;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class MultinomialVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor>, LogProbGraphSupplier {

    private final DoubleVertex p;
    private final IntegerVertex n;
    private static final String P_NAME = "p";
    private static final String N_NAME = "n";

    public MultinomialVertex(@LoadShape long[] tensorShape,
                             @LoadVertexParam(N_NAME) IntegerVertex n,
                             @LoadVertexParam(P_NAME) DoubleVertex p) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, n.getShape());
        long[] pShapeExcludingFirstDimension = TensorShape.removeDimension(0, p.getShape());
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, pShapeExcludingFirstDimension);

        this.p = p;
        this.n = n;

        setParents(p);
        addParent(n);
    }

    @ExportVertexToPythonBindings
    public MultinomialVertex(IntegerVertex n, DoubleVertex p) {
        this(n.getShape(), n, p);
    }

    public MultinomialVertex(int n, DoubleVertex p) {
        this(ConstantVertex.of(IntegerTensor.create(n, new long[]{1, 1})), p);
    }

    @Override
    public double logProb(IntegerTensor k) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).logProb(k).sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceHolderVertex kPlaceholder = new IntegerPlaceHolderVertex(this.getShape());
        IntegerPlaceHolderVertex nPlaceholder = new IntegerPlaceHolderVertex(n.getShape());
        DoublePlaceholderVertex pPlaceholder = new DoublePlaceholderVertex(p.getShape());

        return LogProbGraph.builder()
            .input(this, kPlaceholder)
            .input(n, nPlaceholder)
            .input(p, pPlaceholder)
            .logProbOutput(Multinomial.logProbOutput(kPlaceholder, nPlaceholder, pPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).sample(shape, random);
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
