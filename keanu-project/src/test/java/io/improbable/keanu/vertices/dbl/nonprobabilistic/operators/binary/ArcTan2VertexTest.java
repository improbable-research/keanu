package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class ArcTan2VertexTest {

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void arcTan2OpIsCalculatedCorrectly() {
        ConstantDoubleVertex y = new ConstantDoubleVertex(1.0);
        ConstantDoubleVertex x = new ConstantDoubleVertex(0.0);

        ArcTan2Vertex arcTan2 = new ArcTan2Vertex(y, x);

        assertEquals(Math.atan2(1, 0), arcTan2.getValue(), 0.0001);
    }

    @Test
    public void arcTan2OpIsCalculatedCorrectlyWithValue() {
        ArcTan2Vertex arcTan2 = new ArcTan2Vertex(0, -1);

        assertEquals(Math.atan2(0, -1), arcTan2.getValue(), 0.0001);
    }

    @Test
    public void arcTan2OpIsCalculatedCorrectlyWithValueAsX() {
        ArcTan2Vertex arcTan2 = new ArcTan2Vertex(new ConstantDoubleVertex(0.0), -1);

        assertEquals(Math.atan2(0, -1), arcTan2.getValue(), 0.0001);
    }

    @Test
    public void arcTan2OpIsCalculatedCorrectlyWithValueAsY() {
        ArcTan2Vertex arcTan2 = new ArcTan2Vertex(0, new ConstantDoubleVertex(-1.0));

        assertEquals(Math.atan2(0, -1), arcTan2.getValue(), 0.0001);
    }

    @Test
    public void calculateInfintesimalsWithRespectToAandB() {
        UniformVertex A = new UniformVertex(0, 1, random);
        UniformVertex B = new UniformVertex(0, 1, random);

        A.setValue(0.5);
        double bValue = Math.sqrt(3) / 2.0;
        B.setValue(Math.sqrt(3) / 2.0);

        ArcTan2Vertex arcTan2 = new ArcTan2Vertex(A, B);

        assertEquals(bValue / (Math.pow(bValue, 2) * Math.pow(0.5, 2)), arcTan2.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.001);
        assertEquals(- 0.5 / (Math.pow(bValue, 2) * Math.pow(0.5, 2)), arcTan2.getDualNumber().getInfinitesimal().getInfinitesimals().get(B.getId()), 0.001);
    }

}
