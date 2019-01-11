package io.improbable.keanu.algorithms.variational.optimizer;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    /**
     *
     * @param vertices
     * @return
     */
    double downstreamLogProb(Set<? extends Variable> vertices);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor>> getContinuousLatentVariables();

    void cascadeUpdate(Set<? extends Variable> inputs);

}
