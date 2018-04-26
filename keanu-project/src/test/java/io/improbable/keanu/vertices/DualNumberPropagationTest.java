package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
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
        DoubleVertex start = new SinVertex(Math.PI / 3);

        int links = 20;
        DoubleVertex end = addLinks(start, n, links);

        end.getDualNumber();

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    private DoubleVertex addLinks(DoubleVertex end, AtomicInteger n, int links) {

        for (int i = 0; i < links; i++) {
            DoubleVertex left = passThroughVertex(end, n, id -> log.info("OP on id:" + id));
            DoubleVertex right = passThroughVertex(end, n, id -> log.info("OP on id:" + id));
            end = sumVertex(left, right, n, id -> log.info("OP on id:" + id));
        }

        return end;
    }

    private DoubleVertex passThroughVertex(DoubleVertex from, AtomicInteger n, Consumer<Long> onOp) {
        final long id = Vertex.idGenerator.get();
        return new DoubleUnaryOpLambda<>(from, (a) -> {
            onOp.accept(id);
            return a;
        }, (a) -> {
            n.incrementAndGet();
            return a.get(from);
        });
    }

    private DoubleVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger n, Consumer<Long> onOp) {
        final long id = Vertex.idGenerator.get();
        return new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
            onOp.accept(id);
            return a + b;
        }, (a) -> {
            n.incrementAndGet();
            return a.get(left).add(a.get(right));
        } );
    }

}
