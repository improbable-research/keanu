package io.improbable.keanu.vertices.intgr.probabilistic;

import com.google.common.base.Preconditions;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.TensorShape;
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

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkIsBroadcastable;
import static io.improbable.keanu.tensor.TensorShapeValidation.isBroadcastable;

public class MultinomialVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private final DoubleVertex p;
    private final IntegerVertex n;
    private static final String P_NAME = "p";
    private static final String N_NAME = "n";

    public MultinomialVertex(@LoadShape long[] tensorShape,
                             @LoadVertexParam(N_NAME) IntegerVertex n,
                             @LoadVertexParam(P_NAME) DoubleVertex p) {
        super(tensorShape);

        long[] expectedShape = calculateExpectedShape(n.getShape(), p.getShape());
        Preconditions.checkArgument(Arrays.equals(expectedShape, tensorShape));

        this.p = p;
        this.n = n;

        setParents(p, n);
    }

    private static long[] calculateExpectedShape(long[] nShape, long[] pShape) {
        int pRank = pShape.length;
        long k = pShape[pRank - 1];
        Preconditions.checkArgument(k >= 2, "K value of " + k + " must be greater than 1");

        long[] pBatchShape = TensorShape.selectDimensions(0, pRank - 1, pShape);

        if (!isBroadcastable(nShape, pBatchShape)) {
            throw new IllegalArgumentException(
                "The shape of n " +
                    Arrays.toString(nShape) +
                    " must be broadcastable with the shape of p excluding the k dimension " +
                    Arrays.toString(pBatchShape)
            );
        }

        return TensorShape.concat(checkIsBroadcastable(nShape, pBatchShape), new long[]{k});
    }

    @ExportVertexToPythonBindings
    public MultinomialVertex(IntegerVertex n, DoubleVertex p) {
        this(calculateExpectedShape(n.getShape(), p.getShape()), n, p);
    }

    public MultinomialVertex(int n, DoubleVertex p) {
        this(p.getShape(), ConstantVertex.of(IntegerTensor.scalar(n)), p);
    }

    public MultinomialVertex(int n, DoubleTensor p) {
        this(p.getShape(), ConstantVertex.of(IntegerTensor.scalar(n)), ConstantVertex.of(p));
    }

    public MultinomialVertex(IntegerTensor n, DoubleVertex p) {
        this(ConstantVertex.of(n), p);
    }

    public MultinomialVertex(IntegerTensor n, DoubleTensor p) {
        this(ConstantVertex.of(n), ConstantVertex.of(p));
    }

    @Override
    public double logProb(IntegerTensor xTensor) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).logProb(xTensor).sum();
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