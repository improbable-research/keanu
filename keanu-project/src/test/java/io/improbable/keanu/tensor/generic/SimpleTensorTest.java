package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SimpleTensorTest {

    enum Something {
        A, B, C, D
    }

    @Test
    public void canGetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));

        somethingTensor.setValue(Something.A, 1, 1);

        assertEquals(Something.A, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canReshape() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        Tensor<Something> reshapedSomething = somethingTensor.reshape(9, 1);

        assertArrayEquals(new long[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(somethingTensor.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canTake() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        ConstantGenericVertex<Something> somethingVertex = new ConstantGenericVertex(somethingTensor);

        GenericTakeVertex<Something> take = new GenericTakeVertex(somethingVertex, 0, 0);

        assertEquals(Something.A, take.getValue().scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        Tensor<Something> taddedSomethingRow = somethingTensor.slice(0, 1);
        assertArrayEquals(new long[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = somethingTensor.slice(1, 1);
        assertArrayEquals(new long[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

}
