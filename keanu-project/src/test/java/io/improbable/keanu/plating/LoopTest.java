package io.improbable.keanu.plating;


import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.Test;

import io.improbable.keanu.plating.loop.Loop;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class LoopTest {

    @Test
    public void youCanGetTheOutputVertex() throws VertexLabelException {
        Vertex startValue = ConstantVertex.of(0.);
        Function<DoubleVertex, DoubleVertex> increment = (v) -> v.plus(1.);
        Supplier<BoolVertex> flip = () -> new BernoulliVertex(0.5);
        Loop loop = Loop.startingFrom(startValue).apply(increment).whilst(flip);
        DoubleVertex output = loop.getOutput();
    }


}
