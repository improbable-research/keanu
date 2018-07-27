package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericPluckVertex;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SimpleTensorTest {

    enum Something {
        A, B, C, D
    }

    @Test
    public void whatDoesSliceDo() {
        double[] data = new double[]{
            1, 2, 3, 4, 5, 6, 7, 8,
            5, 6, 7, 8, 9, 10, 11, 12,
            8, 7, 6, 5, 4, 3, 2, 1,
            4, 3, 2, 1, 0, -1, -2, -3
        };

        int[] shape = new int[]{2, 2, 2, 2, 2};
        DataBuffer buffer = Nd4j.createBuffer(data);
        INDArray rank4 = Nd4j.create(buffer, shape);

        INDArray plucked = pluck(rank4, 0, 1);
        assertArrayEquals(new int[]{2, 2, 2}, plucked.shape());
        assertArrayEquals(new double[]{5, 6, 7, 8, 9, 10, 11, 12}, plucked.data().asDouble(), 0.0);
    }


    private INDArray pluck(INDArray from, int... indices) {

        int[] fromShape = from.shape();
        int[] subFromShape = Arrays.copyOf(fromShape, indices.length);
        int pluckIndex = TensorShape.getFlatIndex(subFromShape, TensorShape.getRowFirstStride(subFromShape), indices);
        int[] pluckShape = Arrays.copyOfRange(fromShape, indices.length, fromShape.length);
        int subShapeLength = (int) TensorShape.getLength(subFromShape);

        return from.reshape(subShapeLength, -1)
            .slice(pluckIndex)
            .reshape(pluckShape);
    }

    @Test
    public void canGetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
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
            new int[]{3, 3}
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
            new int[]{3, 3}
        );

        Tensor<Something> reshapedSomething = somethingTensor.reshape(9, 1);

        assertArrayEquals(new int[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(somethingTensor.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canPluck() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        ConstantGenericVertex<Something> somethingVertex = new ConstantGenericVertex(somethingTensor);

        GenericPluckVertex<Something> pluck = new GenericPluckVertex(somethingVertex, 0, 0);

        assertEquals(Something.A, pluck.getValue().scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        Tensor<Something> taddedSomethingRow = somethingTensor.slice(0, 1);
        assertArrayEquals(new int[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = somethingTensor.slice(1, 1);
        assertArrayEquals(new int[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

}
