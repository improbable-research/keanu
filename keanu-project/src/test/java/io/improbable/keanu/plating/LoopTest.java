package io.improbable.keanu.plating;


import java.util.function.Supplier;

import org.junit.Test;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;

public class LoopTest {
    @Test
    public void youCanLoopUntilAConditionIsTrue() throws VertexLabelException {
        Supplier<BoolVertex> condition = () -> new BernoulliVertex(0.5);
        BayesianNetwork loop = Loop
            .startingFrom(ConstantVertex.of(0.))
            .apply(v -> v.plus(ConstantVertex.of(1.)))
            .whilst(condition);

        double logOfMasterP = loop.getLogOfMasterP();
    }

}
