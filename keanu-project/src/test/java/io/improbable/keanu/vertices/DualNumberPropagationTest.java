package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class DualNumberPropagationTest {

    private final Logger log = LoggerFactory.getLogger(DualNumberPropagationTest.class);

    @Test
    public void doesNotPerformUnneccesaryDualNumberCalculations() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(5.0);
        PowerVertex xPow = new PowerVertex(x, 2.0);
        PowerVertex yPow = new PowerVertex(x, 3.0);
        MultiplicationVertex multiplicationVertex = new MultiplicationVertex(xPow, yPow);
        PowerVertex xPow2 = new PowerVertex(multiplicationVertex, 2.0);
        PowerVertex yPow2 = new PowerVertex(multiplicationVertex, 3.0);
        MultiplicationVertex multiplicationVertex1 = new MultiplicationVertex(xPow2, yPow2);
        multiplicationVertex1.getDualNumber();
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
            n.incrementAndGet();
            onOp.accept(id);
            return a;
        });
    }

    private DoubleVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger n, Consumer<Long> onOp) {
        final long id = Vertex.idGenerator.get();
        return new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
            n.incrementAndGet();
            onOp.accept(id);
            return a + b;
        });
    }

}
