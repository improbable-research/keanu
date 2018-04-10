package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PowerVertexTest {

    @Test
    public void calculateSquareCorrectly() {
        DoubleVertex A = new ConstantDoubleVertex(5.);
        DoubleVertex ASquared = new PowerVertex(A, 2.);

        assertEquals(Math.pow(5., 2.), ASquared.lazyEval(), 0.0001);
    }

    @Test
    public void calculateSquareRootCorrectly() {
        DoubleVertex A = new ConstantDoubleVertex(25.);
        DoubleVertex ASquareRoot = new PowerVertex(A, 0.5);

        assertEquals(Math.sqrt(25.), ASquareRoot.lazyEval(), 0.0001);
    }

    @Test
    public void calculateSquareDualNumberCorrectly() {
        DoubleVertex A = new ConstantDoubleVertex(5.);
        DoubleVertex ASquared = new PowerVertex(A, 2.);

        assertEquals(Math.pow(5., 2.), ASquared.getDualNumber().getValue(), 0.0001);
    }

    @Test
    public void calculateInfintesimal() {
        DoubleVertex A = new UniformVertex(0, 10);
        A.setValue(4d);
        DoubleVertex B = new LogVertex(A);
        //Differential of B = 1 / A
        DoubleVertex APower = new PowerVertex(B, 3.);
        //Differential of APower = (3 - 1) * ((1 / A) ^ 3) * (1 / A)
        assertEquals((Math.pow(Math.log(4), 2)) * 3.0 / 4.0, APower.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.0001);
    }

}
