package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Arrays;

public abstract class BinaryOpVertex<A, B, C> extends Vertex<C> implements NonProbabilistic<C> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public C sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public C calculate() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract C op(A a, B b);

    private static boolean shouldCorrectPartialForScalarForward(PartialDerivative partial, long[] targetOfShape, long[] currentOfShape) {
        return partial.isPresent() && !Arrays.equals(currentOfShape, targetOfShape);
    }

    public static PartialDerivative correctForScalarPartialForward(PartialDerivative partialDerivative, long[] targetOfShape, long[] currentOfShape) {

        if (shouldCorrectPartialForScalarForward(partialDerivative, currentOfShape, targetOfShape)) {

            DoubleTensor partial = partialDerivative.getPartial();
            long[] partialShape = partial.getShape();
            long[] wrtShape = TensorShape.selectDimensions(currentOfShape.length, partialShape.length, partialShape);
            DoubleTensor correctedPartial = DoubleTensor.zeros(TensorShape.concat(targetOfShape, wrtShape)).plus(partial);
            return new PartialDerivative(partialDerivative.getKey(), correctedPartial);
        } else {
            return partialDerivative;
        }
    }

    private static boolean shouldCorrectForPartialScalarReverse(PartialDerivative partial, long[] targetWrtShape, long[] currentWrtShape) {
        return partial.isPresent() && !Arrays.equals(currentWrtShape, targetWrtShape);
    }

    public static PartialDerivative correctForScalarReverse(PartialDerivative partialForScalar, long[] currentWrtShape, long[] targetWrtShape) {

        if (shouldCorrectForPartialScalarReverse(partialForScalar, currentWrtShape, targetWrtShape)) {

            long[] partialShape = partialForScalar.getPartial().getShape();
            int[] wrtDims = TensorShape.dimensionRange(partialShape.length - currentWrtShape.length, partialShape.length);

            return partialForScalar.sumOverWrtDimensions(wrtDims, targetWrtShape);
        } else {
            return partialForScalar;
        }
    }
}
