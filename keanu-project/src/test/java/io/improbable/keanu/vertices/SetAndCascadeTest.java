package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

public class SetAndCascadeTest {

    @Test
    public void doesNotDoUnnecessaryOperations() {

        Random random = new Random(1);
        DoubleVertex start = new GaussianVertex(0, 1, random);

        DoubleVertex end = start;

        AtomicInteger n = new AtomicInteger(0);

        int links = 20;
        for (int i = 0; i < links; i++) {

            DoubleVertex left = new DoubleUnaryOpLambda<>(end, (a) -> {
                n.incrementAndGet();
                return a;
            });

            DoubleVertex right = new DoubleUnaryOpLambda<>(end, (a) -> {
                n.incrementAndGet();
                return a;
            });

            end = new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
                n.incrementAndGet();
                return a + b;
            });
        }

        start.setAndCascade(2.0);

        //Calculates the correct answer
        assertEquals(end.getValue(), Math.pow(2, links + 1), 0.0);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }
}
