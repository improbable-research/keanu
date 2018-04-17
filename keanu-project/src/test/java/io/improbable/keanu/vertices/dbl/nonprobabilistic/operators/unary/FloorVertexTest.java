package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class FloorVertexTest {

    @Test
    public void canComputeFloorOperation() {
        DoubleVertex A = new ConstantDoubleVertex(8.12);
        FloorVertex floor = new FloorVertex(A);

        assertEquals(Math.floor(8.12), floor.getValue(), 0.0001);
    }

    @Test
    public void canComputeFloorOperationWithValue() {
        FloorVertex floor = new FloorVertex(8.12);

        assertEquals(Math.floor(8.12), floor.getValue(), 0.0001);
    }

    @Test
    public void floorOnlyChangesValueAndNotInfintesimal() {
        DoubleVertex A = new UniformVertex(0.0, 10.0);
        A.setValue(5.2);
        FloorVertex floorVertex = new FloorVertex(A);
        PowerVertex powerVertex = new PowerVertex(floorVertex, 2.0);

        assertEquals(2 * Math.pow(5.0, 2 - 1), powerVertex.getDualNumber().getInfinitesimal().getInfinitesimals().get(A.getId()), 0.000);
    }

}
