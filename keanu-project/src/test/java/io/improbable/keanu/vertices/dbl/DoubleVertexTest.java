package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.function.Function;

import static org.junit.Assert.assertTrue;

public class DoubleVertexTest {

    DoubleVertex v1;
    DoubleVertex v2;

    @Before
    public void setup() {
        v1 = new GaussianVertex(0.0, 1.0);
        v2 = new GaussianVertex(0.0, 1.0);

        v1.setValue(1.0);
        v2.setValue(2.0);
    }

    @Test
    public void doesMultiply() {
        DoubleVertex v3 = v1.multiply(v2);
        v3.lazyEval();
        assertTrue(v3.getValue() == 2.0);
    }

    @Test
    public void doesAdd() {
        DoubleVertex v3 = v1.plus(v2);
        v3.lazyEval();
        assertTrue(v3.getValue() == 3.0);
    }

    @Test
    public void doesSubtract() {
        DoubleVertex v3 = v1.minus(v2);
        v3.lazyEval();
        assertTrue(v3.getValue() == -1.0);
    }

    @Test
    public void doesPower() {
        DoubleVertex v3 = v2.pow(3);
        v3.lazyEval();
        assertTrue(v3.getValue() == 8.0);
    }

    @Test
    public void doesLambda() {

        Function<Double, Double> op = val -> val + 5;

        DoubleVertex v3 = v1.lambda(op, (a) -> {
            DualNumber v1Dual = v1.getDualNumber();
            return new DualNumber(op.apply(v1Dual.getValue()), v1Dual.getPartialDerivatives());
        });

        v3.lazyEval();
        assertTrue(v3.getValue() == 6);

        DualNumber v3Dual = v3.getDualNumber();
        assertTrue(v3Dual.getValue() == 6);
    }

}
