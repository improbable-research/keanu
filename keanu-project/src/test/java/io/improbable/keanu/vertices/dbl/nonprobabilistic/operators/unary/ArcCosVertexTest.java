package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ArcCosVertexTest {

    @Test
    public void arcCosOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(0.5);
        ArcCosVertex aCos = new ArcCosVertex(x);

        assertEquals(Math.acos(0.5), aCos.getValue(), 0.00001);
    }

    @Test
    public void arcCosOpIsCalculatedCorrectlyWithValue() {
        ArcCosVertex aCosWithValue = new ArcCosVertex(0.5);

        assertEquals(Math.acos(0.5), aCosWithValue.getValue(), 0.00001);
    }

    @Test
    public void arcCosDualNumberIsCalculatedCorrectly() {
        DoubleVertex uniform = new UniformVertex(0, 10);
        uniform.setValue(5.0);

        DoubleVertex pow = new PowerVertex(uniform, 3); //dPow = 3 * 5^2
        ArcCosVertex aCos = new ArcCosVertex(pow);

        double dArcCos = aCos.getDualNumber().getInfinitesimal().getInfinitesimals().get(uniform.getId());
        //dArcCos = π / 2 - asin(5^3) * 3 * 5^2
        double expected = Math.PI / 2 - Math.asin(Math.pow(uniform.getValue(), 3)) * (3 * Math.pow(uniform.getValue(), 3 - 1));

        assertEquals(expected, dArcCos, 0.0001);
    }

}
