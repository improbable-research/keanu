package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.TestGraphGenerator;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static junit.framework.TestCase.assertEquals;

public class DifferentiatorEfficiencyTest {

    private UniformVertex start;
    private TestGraphGenerator.SumVertex end;

    private AtomicInteger n;
    private AtomicInteger m;
    private int links = 20;

    @Before
    public void setup() {
        n = new AtomicInteger(0);
        m = new AtomicInteger(0);
        start = new UniformVertex(0, 1);
        end = TestGraphGenerator.addLinks(start, n, m, links);
    }

    @Test
    public void doesNotPerformUnnecessaryReverseModeAutoDiffCalculations() {
        Differentiator.reverseModeAutoDiff(end, start);
        assertEquals(3 * links, m.get());
    }

    @Test
    public void doesNotPerformUnnecessaryForwardModeAutoDiffCalculations() {
        Differentiator.forwardModeAutoDiff(start, end);
        assertEquals(3 * links, m.get());
    }
}
