package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;

import java.util.HashMap;
import java.util.Map;

/**
 * This class stores the gradients of a log probability. It serves
 * as way to sum multiple gradients from the same vertices.
 */
public class LogProbGradients {

    private final Map<VertexId, DoubleTensor> partials;

    public LogProbGradients() {
        this.partials = new HashMap<>();
    }

    public LogProbGradients add(LogProbGradients addition) {
        return add(addition.partials);
    }

    public LogProbGradients add(Map<VertexId, DoubleTensor> addition) {

        for (Map.Entry<VertexId, DoubleTensor> entry : addition.entrySet()) {

            VertexId id = entry.getKey();
            DoubleTensor existingPartialDerivative = partials.get(id);

            if (existingPartialDerivative == null) {
                partials.put(id, entry.getValue().duplicate());
            } else {
                partials.put(id, existingPartialDerivative.plusInPlace(entry.getValue()));
            }
        }

        return this;
    }

    public DoubleTensor getWithRespectTo(VertexId id) {
        return this.partials.get(id);
    }

    public void putWithRespectTo(VertexId id, DoubleTensor partial) {
        this.partials.put(id, partial);
    }

    public Map<VertexId, DoubleTensor> getPartials() {
        return partials;
    }
}
