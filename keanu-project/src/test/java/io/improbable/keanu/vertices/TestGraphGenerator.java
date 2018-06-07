package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class TestGraphGenerator {

    private static final Logger log = LoggerFactory.getLogger(TestGraphGenerator.class);

    static DoubleVertex addLinks(DoubleVertex end, AtomicInteger opCount, AtomicInteger dualNumberCount, int links) {

        for (int i = 0; i < links; i++) {
            DoubleVertex left = passThroughVertex(end, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
            DoubleVertex right = passThroughVertex(end, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
            end = sumVertex(left, right, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
        }

        return end;
    }

    static DoubleVertex passThroughVertex(DoubleVertex from, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        final long id = Vertex.ID_GENERATOR.get();
        return new DoubleUnaryOpLambda<>(from, (a) -> {
            opCount.incrementAndGet();
            onOp.accept(id);
            return a;
        }, (a) -> {
            dualNumberCount.incrementAndGet();
            return a.get(from);
        });
    }

    static DoubleVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        final long id = Vertex.ID_GENERATOR.get();
        return new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
            opCount.incrementAndGet();
            onOp.accept(id);
            return a.plus(b);
        }, (a) -> {
            dualNumberCount.incrementAndGet();
            return a.get(left).add(a.get(right));
        });
    }

}
