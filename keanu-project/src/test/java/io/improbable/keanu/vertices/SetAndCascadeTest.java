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
        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1, random);

        int links = 20;
        DoubleVertex end = addLinks(start, n, links);

        start.setAndCascade(2.0);

        //Calculates the correct answer
        assertEquals(Math.pow(2, links + 1), end.getValue(), 0.0);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVertices() {
        Random random = new Random(1);
        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1, random);

        DoubleVertex end = addLinks(start, n, 1);

        start.setAndCascade(2.0);

        System.out.println("");
        System.out.println("");
        DoubleVertex nextLayerStart = new GaussianVertex(end, 1, random);

        DoubleVertex secondLayerEnd = addLinks(nextLayerStart, n, 1);

        start.setAndCascade(3.0);

        assertEquals(6.0, end.getValue(), 0.0);
        assertEquals(3, n.get());
    }

    private DoubleVertex addLinks(DoubleVertex end, AtomicInteger n, int links) {


        long id;

        for (int i = 0; i < links; i++) {

            final long leftId = Vertex.idGenerator.get();
            DoubleVertex left = new DoubleUnaryOpLambda<>(end, (a) -> {
                n.incrementAndGet();
                System.out.println("left " + leftId + " " + n.get());
                return a;
            });

            final long rightId = Vertex.idGenerator.get();
            DoubleVertex right = new DoubleUnaryOpLambda<>(end, (a) -> {
                n.incrementAndGet();
                System.out.println("right " + rightId + " " + n.get());
                return a;
            });

            final long centerId = Vertex.idGenerator.get();
            end = new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
                n.incrementAndGet();
                System.out.println("center " + centerId + " " + n.get());
                return a + b;
            });
        }

        return end;
    }
}
