package io.improbable.keanu.vertices;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;

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

    static class PassThroughVertex extends DoubleUnaryOpVertex {

        private final long id;
        private final AtomicInteger opCount;
        private final AtomicInteger dualNumberCount;
        private final Consumer<Long> onOp;

        public PassThroughVertex(DoubleVertex inputVertex, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
            super(inputVertex);
            this.opCount = opCount;
            this.dualNumberCount = dualNumberCount;
            this.onOp = onOp;
            id = Vertex.ID_GENERATOR.get();
        }

        @Override
        protected DoubleTensor op(DoubleTensor a) {
            opCount.incrementAndGet();
            onOp.accept(id);
            return a;
        }

        @Override
        protected DualNumber dualOp(DualNumber a) {
            dualNumberCount.incrementAndGet();
            return a;
        }
    }

    static DoubleVertex passThroughVertex(DoubleVertex from, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        return new PassThroughVertex(from, opCount, dualNumberCount, onOp);
    }

    static class SumVertex extends DoubleBinaryOpVertex {

        private final AtomicInteger opCount;
        private final AtomicInteger dualNumberCount;
        private final Consumer<Long> onOp;
        private final long id;

        public SumVertex(DoubleVertex left, DoubleVertex right,
                         AtomicInteger opCount, AtomicInteger dualNumberCount,
                         Consumer<Long> onOp) {
            super(left, right);
            this.opCount = opCount;
            this.dualNumberCount = dualNumberCount;
            this.onOp = onOp;
            id = Vertex.ID_GENERATOR.get();
        }

        @Override
        protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
            opCount.incrementAndGet();
            onOp.accept(id);
            return l.plus(r);
        }

        @Override
        protected DualNumber dualOp(DualNumber l, DualNumber r) {
            dualNumberCount.incrementAndGet();
            return l.add(r);
        }
    }

    static DoubleVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        return new SumVertex(left, right, opCount, dualNumberCount, onOp);
    }

}
