package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;

public class DualNumberPropagationTest {

    private final Logger log = LoggerFactory.getLogger(DualNumberPropagationTest.class);

    @Test
    public void doesNotPerformUnneccesaryDualNumberCalculations() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = new SinVertex(Math.PI / 3);

        int links = 20;
        DoubleVertex end = TestGraphGenerator.addLinks(start, n, m, links);

        end.getDualNumber();

        //Does the right amount of work
        assertEquals(3 * links, m.get());
    }

}
