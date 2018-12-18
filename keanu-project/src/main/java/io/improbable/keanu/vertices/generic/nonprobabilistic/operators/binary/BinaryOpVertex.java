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

    public static boolean shouldCorrectPartialForScalar(PartialDerivative dSideWrtInput, long[] opShape, long[] sideShape) {
        return dSideWrtInput.isPresent() && !Arrays.equals(sideShape, opShape);
    }

    public static PartialDerivative correctForScalarPartial(PartialDerivative partialDerivative, long[] opShape, int sideRank) {
        DoubleTensor partial = partialDerivative.getPartial();
        long[] partialShape = partial.getShape();
        long[] wrtShape = TensorShape.selectDimensions(sideRank, partialShape.length, partialShape);
        DoubleTensor correctedPartial = DoubleTensor.zeros(TensorShape.concat(opShape, wrtShape)).plus(partial);
        return new PartialDerivative(partialDerivative.getKey(), correctedPartial);
    }
}
