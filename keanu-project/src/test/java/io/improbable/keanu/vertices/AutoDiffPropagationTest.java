package io.improbable.keanu.vertices;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class AutoDiffPropagationTest {

    @Test
    public void doesNotPerformUnneccesaryAutoDiffCalculations() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = ConstantVertex.of(Math.PI / 3).sin();

        int links = 20;
        DoubleVertex end = TestGraphGenerator.addLinks(start, n, m, links);

        end.getDerivativeWrtLatents();

        //Does the right amount of work
        assertEquals(3 * links, m.get());
    }

}
