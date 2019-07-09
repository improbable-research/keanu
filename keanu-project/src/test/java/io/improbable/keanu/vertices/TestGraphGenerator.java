package io.improbable.keanu.vertices;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import lombok.extern.slf4j.Slf4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

@Slf4j
public class TestGraphGenerator {

    public static SumVertex addLinks(DoubleVertex end, AtomicInteger opCount, AtomicInteger autoDiffCount, int links) {
        Preconditions.checkArgument(links > 0);

        for (int i = 0; i < links; i++) {
            DoubleVertex left = passThroughVertex(end, opCount, autoDiffCount, id -> log.info("OP on id:" + id));
            DoubleVertex right = passThroughVertex(end, opCount, autoDiffCount, id -> log.info("OP on id:" + id));
            end = sumVertex(left, right, opCount, autoDiffCount, id -> log.info("OP on id:" + id));
        }

        return (SumVertex) end;
    }

    public static class PassThroughVertex extends DoubleUnaryOpVertex implements Differentiable, NonSaveableVertex {

        private final AtomicInteger opCount;
        private final AtomicInteger autoDiffCount;
        private final Consumer<VertexId> onOp;

        public PassThroughVertex(DoubleVertex inputVertex, AtomicInteger opCount, AtomicInteger autoDiffCount, Consumer<VertexId> onOp) {
            super(inputVertex);
            this.opCount = opCount;
            this.autoDiffCount = autoDiffCount;
            this.onOp = onOp;
        }

        @Override
        protected DoubleTensor op(DoubleTensor a) {
            opCount.incrementAndGet();
            onOp.accept(getId());
            return a;
        }

        @Override
        public PartialDerivative forwardModeAutoDifferentiation(Map<IVertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
            PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
            autoDiffCount.incrementAndGet();
            return derivativeOfParentWithRespectToInputs;
        }

        @Override
        public Map<IVertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
            autoDiffCount.incrementAndGet();
            return Collections.singletonMap(inputVertex, derivativeOfOutputWithRespectToSelf);
        }
    }

    public static DoubleVertex passThroughVertex(DoubleVertex from, AtomicInteger opCount, AtomicInteger autoDiffCount, Consumer<VertexId> onOp) {
        return new PassThroughVertex(from, opCount, autoDiffCount, onOp);
    }

    public static class SumVertex extends DoubleBinaryOpVertex implements Differentiable, NonSaveableVertex {

        private final AtomicInteger opCount;
        private final AtomicInteger autoDiffCount;
        private final Consumer<VertexId> onOp;

        public SumVertex(DoubleVertex left, DoubleVertex right,
                         AtomicInteger opCount, AtomicInteger autoDiffCount,
                         Consumer<VertexId> onOp) {
            super(left, right);
            this.opCount = opCount;
            this.autoDiffCount = autoDiffCount;
            this.onOp = onOp;
        }

        public SumVertex(@LoadVertexParam("left") DoubleVertex left, @LoadVertexParam("right") DoubleVertex right) {
            super(left, right);
            this.opCount = new AtomicInteger();
            this.autoDiffCount = new AtomicInteger();
            this.onOp = null;
        }

        @Override
        protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
            opCount.incrementAndGet();
            onOp.accept(getId());
            return l.plus(r);
        }

        @Override
        public PartialDerivative forwardModeAutoDifferentiation(Map<IVertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
            PartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
            PartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);
            autoDiffCount.incrementAndGet();
            return dLeftWrtInput.add(dRightWrtInput);
        }

        @Override
        public Map<IVertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
            autoDiffCount.incrementAndGet();
            Map<IVertex, PartialDerivative> partials = new HashMap<>();
            partials.put(left, derivativeOfOutputWithRespectToSelf);
            partials.put(right, derivativeOfOutputWithRespectToSelf);
            return partials;
        }
    }

    public static SumVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<VertexId> onOp) {
        return new SumVertex(left, right, opCount, dualNumberCount, onOp);
    }

}
