package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ArcSinVertexTest {

    @Test
    public void arcSineOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(0.5);
        ArcSinVertex arcSin = new ArcSinVertex(x);

        assertEquals(Math.asin(0.5), arcSin.getValue(), 0.00001);
    }

    @Test
    public void arcSineOpIsCalculatedCorrectlyWithValues() {
        ArcSinVertex arcSinWithValue = new ArcSinVertex(0.5);

        assertEquals(Math.asin(0.5), arcSinWithValue.getValue(), 0.00001);
    }

    @Test
    public void sinDualNumberIsCalculatedCorrectly() {
        DoubleVertex uniform = new UniformVertex(0, 1);
        uniform.setValue(0.5);
        //dPow = 3 * 0.5^2
        DoubleVertex pow = new PowerVertex(uniform, 3);

        ArcSinVertex aSine = new ArcSinVertex(pow);
        double dArcSine = aSine.getDualNumber().getInfinitesimal().getInfinitesimals().get(uniform.getId());
        //dArcSine = 1 / √(1 - (0.5^3)^2) * 3 * 0.5^2
        double expected = 1 / Math.sqrt(1 - Math.pow(Math.pow(uniform.getValue(), 3), 2)) * (3 * Math.pow(uniform.getValue(), 3 - 1));

        assertEquals(expected, dArcSine, 0.0001);
    }

}