package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SinVertexTest {

    @Test
    public void sinOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(Math.PI / 2);
        SinVertex sin = new SinVertex(x);

        assertEquals(Math.sin(Math.PI / 2), sin.getValue(), 0.00001);
    }

    @Test
    public void sinOpIsCalculatedCorrectlyWithValue() {
        SinVertex sinWithValue = new SinVertex(Math.PI / 2);

        assertEquals(Math.sin(Math.PI / 2), sinWithValue.getValue(), 0.00001);
    }

    @Test
    public void sinDualNumberIsCalculatedCorrectly() {
        DoubleVertex uniform = new UniformVertex(0, 10);
        uniform.setValue(5.0);

        DoubleVertex pow = new PowerVertex(uniform, 3); //dPow = 3 * 5^2
        SinVertex sin = new SinVertex(pow);

        double dSin = sin.getDualNumber().getInfinitesimal().getInfinitesimals().get(uniform.getId());
        //dSin = cos(5^3) * 3 * 5^2
        double expected = Math.cos(Math.pow(uniform.getValue(), 3)) * (3 * Math.pow(uniform.getValue(), 3 - 1));

        assertEquals(expected, dSin, 0.0001);
    }

}
