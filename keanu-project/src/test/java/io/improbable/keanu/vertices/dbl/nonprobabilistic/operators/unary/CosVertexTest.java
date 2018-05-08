package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CosVertexTest {

    @Test
    public void cosOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(Math.PI / 2);
        CosVertex cos = new CosVertex(x);

        assertEquals(Math.cos(Math.PI / 2), cos.getValue(), 0.00001);
    }

    @Test
    public void cosOpIsCalculatedCorrectlyWithValue() {
        CosVertex cosWithValue = new CosVertex(Math.PI / 2);

        assertEquals(Math.cos(Math.PI / 2), cosWithValue.getValue(), 0.00001);
    }

    @Test
    public void cosDualNumberIsCalculatedCorrectly() {
        DoubleVertex uniform = new UniformVertex(0, 10);
        uniform.setValue(5.0);

        DoubleVertex pow = new PowerVertex(uniform, 3); //dPow = 3 * 5^2
        CosVertex cos = new CosVertex(pow);

        double dCos = cos.getDualNumber().getPartialDerivatives().withRespectTo(uniform);
        //dCos = -sin(5^3) * 3 * 5^2
        double expected = -Math.sin(Math.pow(uniform.getValue(), 3)) * (3 * Math.pow(uniform.getValue(), 3 - 1));

        assertEquals(expected, dCos, 0.0001);
    }

}
