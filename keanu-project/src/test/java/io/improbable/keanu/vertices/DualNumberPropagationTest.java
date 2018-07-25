package io.improbable.keanu.vertices;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiator;

public class DualNumberPropagationTest {

    private final Logger log = LoggerFactory.getLogger(DualNumberPropagationTest.class);

    @Test
    public void doesNotPerformUnneccesaryDualNumberCalculations() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = ConstantVertex.of(Math.PI / 3).sin();

        int links = 20;
        DoubleVertex end = TestGraphGenerator.addLinks(start, n, m, links);

        new Differentiator().calculateDual((Differentiable)end);

        //Does the right amount of work
        assertEquals(3 * links, m.get());
    }

}
