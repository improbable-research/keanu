package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.function.BiFunction;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BooleanBinaryOpVertex<A extends Tensor, B extends Tensor> extends BooleanVertex {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    private final BiFunction<A, B, BooleanTensor> op;

    public BooleanBinaryOpVertex(Vertex<A> a, Vertex<B> b, BiFunction<A, B, BooleanTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op);
    }

    public BooleanBinaryOpVertex(int[] shape, Vertex<A> a, Vertex<B> b, BiFunction<A, B, BooleanTensor> op) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(a.getValue(), b.getValue())),
            Observable.observableTypeFor(BooleanBinaryOpVertex.class)
        );

        this.a = a;
        this.b = b;
        this.op = op;
        setParents(a, b);
        setValue(BooleanTensor.placeHolder(shape));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }

    @Override
    public boolean matchesObservation() {
        return op.apply(a.getValue(), b.getValue()).elementwiseEquals(getValue()).allTrue();
    }
}