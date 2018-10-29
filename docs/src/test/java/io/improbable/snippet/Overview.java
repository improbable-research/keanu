package io.improbable.snippet;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;


public class Overview {

    public static void main(String[] args) {
//%%SNIPPET_START%% Overview
DoubleVertex x = new UniformVertex(1, 2);
DoubleVertex y = x.times(2);
DoubleVertex observedY = new UniformVertex(new long[]{1, 2}, y, y.plus(0.5));
observedY.observe(new double[]{4.0, 4.49});
//%%SNIPPET_END%% Overview
    }
}
