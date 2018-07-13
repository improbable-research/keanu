package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class DoubleBinaryOpLambda<A, B> extends NonProbabilisticDouble implements Differentiable {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation;

    public DoubleBinaryOpLambda(int[] shape,
                                Vertex<A> a,
                                Vertex<B> b,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        super(v -> ((DoubleBinaryOpLambda<A, B>) v).op.apply(a.getValue(), b.getValue()));
        this.a = a;
        this.b = b;
        this.op = op;
        this.dualNumberCalculation = dualNumberCalculation;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleBinaryOpLambda(int[] shape, Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(shape, a, b, op, null);
    }

    public DoubleBinaryOpLambda(Vertex<A> a,
                                Vertex<B> b,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op, dualNumberCalculation);
    }

    public DoubleBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (dualNumberCalculation != null) {
            return dualNumberCalculation.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}


