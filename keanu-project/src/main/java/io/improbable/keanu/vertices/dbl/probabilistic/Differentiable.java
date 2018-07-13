package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public interface Differentiable extends IVertex {
    DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers);
}
