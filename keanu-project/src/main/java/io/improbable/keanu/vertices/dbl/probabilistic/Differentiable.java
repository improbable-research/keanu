package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public interface Differentiable extends IVertex<DoubleTensor, Vertex<?>> {

    DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers);

    default DualNumber getDualNumber() {
        return new Differentiator().calculateDual(this);
    }
}
