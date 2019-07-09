package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.CastNumberToBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.NumericalEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastNumberToDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastNumberToIntegerVertex;

public interface FixedPointTensorVertex<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, VERTEX extends FixedPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFixedPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

}
