package io.improbable.keanu.network;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface NetworkReader {

    void loadValue(DoubleVertex vertex);
    void loadValue(BoolVertex vertex);
    void loadValue(IntegerVertex vertex);

    DoubleTensor getInitialDoubleValue(Object valueKey);
    BooleanTensor getInitialBoolValue(Object valueKey);
    IntegerTensor getInitialIntValue(Object valueKey);
}
