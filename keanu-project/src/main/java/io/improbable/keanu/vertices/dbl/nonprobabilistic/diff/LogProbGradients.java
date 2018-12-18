package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;

import java.util.HashMap;
import java.util.Map;

public class LogProbGradients {

    private final Map<VertexId, DoubleTensor> partials;

    public LogProbGradients() {
        this.partials = new HashMap<>();
    }

    public LogProbGradients add(LogProbGradients addition) {
        return add(addition.partials);
    }

    public LogProbGradients add(Map<VertexId, DoubleTensor> addition) {

        Map<VertexId, DoubleTensor> initial = partials;

        for (Map.Entry<VertexId, DoubleTensor> entry : addition.entrySet()) {
            VertexId id = entry.getKey();
            if (initial.containsKey(id)) {

                DoubleTensor summation = initial.get(entry.getKey()).plus(entry.getValue());

                initial.put(id, summation);
            } else {
                initial.put(id, entry.getValue());
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
