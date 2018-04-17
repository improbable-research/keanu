package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ArcTanVertexTest {


    @Test
    public void arcTanOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(Math.PI / 2);
        ArcTanVertex aTan = new ArcTanVertex(x);

        assertEquals(Math.atan(Math.PI / 2), aTan.getValue(), 0.0001);
    }

    @Test
    public void arcTanOpIsCalculatedCorrectlyWithValue() {
        ArcTanVertex aTan = new ArcTanVertex(Math.PI / 2);

        assertEquals(Math.atan(Math.PI / 2), aTan.getValue(), 0.0001);
    }

    @Test
    public void arcTanDualNumberIsCalculatedCorrectly() {
        UniformVertex uniformVertex = new UniformVertex(0, 10);
        uniformVertex.setValue(5.0);

        ArcTanVertex arcTan = new ArcTanVertex(uniformVertex);

        double dArcTan = arcTan.getDualNumber().getInfinitesimal().getInfinitesimals().get(uniformVertex.getId());
        //dArcTan = 1 / (1 + x^2)
        double expected = 1 / (1 + Math.pow(5.0, 2));

        assertEquals(expected, dArcTan, 0.001);
    }

}
