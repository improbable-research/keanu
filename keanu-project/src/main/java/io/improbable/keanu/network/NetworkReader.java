package io.improbable.keanu.network;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public interface NetworkReader {

    void loadValue(DoubleVertex vertex);

    DoubleTensor getInitialDoubleValue(Object valueKey);
}
