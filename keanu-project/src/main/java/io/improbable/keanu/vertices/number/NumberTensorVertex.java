package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseNumberTensor;
import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.kotlin.NumberOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.CastNumberToBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastNumberToDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastNumberToIntegerVertex;

public interface NumberTensorVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends TensorVertex<T, TENSOR, VERTEX>, BaseNumberTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX>, NumberOperators<VERTEX> {


    @DisplayInformationForOutput(displayName = "-")
    class NumberDifferenceVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
        extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

        @ExportVertexToPythonBindings
        public NumberDifferenceVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                      @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
            super(left, right);
        }

        @Override
        protected TENSOR op(TENSOR l, TENSOR r) {
            return l.minus(r);
        }
    }

    @Override
    default VERTEX minus(VERTEX that) {
        return asTyped(new NumberDifferenceVertex<>(this, that));
    }

    @Override
    default BooleanVertex toBoolean() {
        return new CastNumberToBooleanVertex<>(this);
    }

    @Override
    default DoubleVertex toDouble() {
        return new CastNumberToDoubleVertex<>(this);
    }

    @Override
    default IntegerVertex toInteger() {
        return new CastNumberToIntegerVertex<>(this);
    }

    IntegerVertex argMax(int axis);

    IntegerVertex argMax();

    IntegerVertex argMin(int axis);

    IntegerVertex argMin();

    @Override
    default BooleanVertex equalsWithinEpsilon(VERTEX other, T epsilon) {
        return new NumericalEqualsVertex<>(this, other, epsilon);
    }

    @Override
    default BooleanVertex greaterThan(VERTEX rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex greaterThanOrEqual(VERTEX rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex lessThan(VERTEX rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex lessThanOrEqual(VERTEX rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }
}
