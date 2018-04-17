package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class PowerVertexTest {

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

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
    public void calculatePowerVertexUsingVertexAsHyperAndValueAsBase() {
        DoubleVertex A = new ConstantDoubleVertex(2.0);
        DoubleVertex B = new PowerVertex(5.0, A);

        assertEquals(Math.pow(5, 2), B.getValue(), 0.0001);
    }

    @Test
    public void calculatePowerVertexUsingVertexAsHyper() {
        DoubleVertex A = new ConstantDoubleVertex(5.);
        DoubleVertex B = new GaussianVertex(2.0, 1.0, random);
        B.setValue(2.0);
        DoubleVertex C = A.pow(B);

        assertEquals(Math.pow(5, 2), C.getValue(), 0.0001);
    }

    @Test
    public void calculateInfintesimal() {
        DoubleVertex A = new UniformVertex(0, 10, random);
        A.setValue(4d);
        DoubleVertex B = new LogVertex(A);
        //Differential of B = 1 / A
        DoubleVertex APower = new PowerVertex(B, 3.);
        //Differential of APower = (3 - 1) * ((1 / A) ^ 3) * (1 / A)

        assertEquals((Math.pow(Math.log(4), 2)) * 3.0 / 4.0, APower.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.0001);
    }

    @Test
    public void calculateInfintesimalUsingVertexAsHyper() {
        DoubleVertex A = new UniformVertex(0, 10, random);
        A.setValue(4d);
        DoubleVertex B = new LogVertex(A);
        //Differential of B = 1 / A
        DoubleVertex APower = new PowerVertex(B, new ConstantDoubleVertex(3.0));
        //Differential of APower = (3 - 1) * ((1 / A) ^ 3) * (1 / A)

        assertEquals((Math.pow(Math.log(4), 2)) * 3.0 / 4.0, APower.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.0001);
    }

    @Test
    public void calculateInfintesimalsWithRespectToAandB() {
        UniformVertex A = new UniformVertex(1.0, 5.0, random);
        UniformVertex B = new UniformVertex(1.0, 10.0, random);

        A.setValue(4.0);
        B.setValue(3.0);

        PowerVertex P = new PowerVertex(A, B);

        Assert.assertEquals((3 * Math.pow(4, 3 - 1)), P.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.001);
        Assert.assertEquals(Math.pow(4, 3) * Math.log(4), P.getDualNumber().getInfinitesimal().getInfinitesimals().get(B.getId()), 0.001);
    }

}
