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
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.isBroadcastable;

/**
 * The multinomial vertex is a multivariate distribution with a shape determined by p (probabilities) and n (trials).
 * It does support batch sampling and batch logProb.
 * <p>
 * The most common use case is a single scalar value for n (trials) and a vector of p (probabilities):
 * e.g.
 * n = 5 with shape ()
 * p = [0.2, 0.2, 0.6] with shape (3)
 * a sample could return x = [1, 3, 1] with shape (3)
 * and logProb([1, 3, 1]) would be valid
 * and logProb([[1, 3, 1], [2, 2, 1]]) would be a batch logProb equivalent to logProb([1, 3, 1]) + logProb([2, 2, 1])
 * <p>
 * More complex cases are also acceptable and use broadcasting semantics.
 * <p>
 * If the number of categories is defined by k, then the shape of p is (a...b, k) where a...b represents any shape of
 * any rank. For the p as a vector case, a...b is rank 0 and would be just a shape (k). Given that p has a shape of
 * (a...b, k) then n can have any shape that is broadcastable with a...b. The resulting shape would be the broadcasted
 * n shape with a...b and end in k.
 * e.g.
 * n = [[1, 2],[3, 4]] with shape (2, 2)
 * p = [[0.2, 0.2, 0.6], [0.5, 0.25, 0.25]] with shape 2, 3
 * therefore k = 3
 * and the result shape is (2, 2, 3), which is (2, 2) broadcasted with (2) and k appended.
 */
public class MultinomialVertex extends VertexImpl<IntegerTensor> implements IntegerVertex,   ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private final DoubleVertex p;
    private final IntegerVertex n;

    @Getter
    @Setter
    private boolean validationEnabled;

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
        this.validationEnabled = true;

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

        return TensorShape.concat(TensorShape.getBroadcastResultShape(nShape, pBatchShape), new long[]{k});
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
        return Multinomial.withParameters(n.getValue(), p.getValue(), validationEnabled).logProb(xTensor).sumNumber();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Multinomial.withParameters(n.getValue(), p.getValue(), validationEnabled).sample(shape, random);
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