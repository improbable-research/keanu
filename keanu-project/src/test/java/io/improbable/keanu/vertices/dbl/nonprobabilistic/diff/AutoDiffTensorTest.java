package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class AutoDiffTensorTest {

    @Test
    public void diffWrtVectorOverMultipleMultiplies() {

        DoubleVertex A = new UniformVertex(new int[]{1, 4}, 0, 1);
        A.setValue(new double[]{-1, 3, 5, -2});

        DoubleVertex prod = A.times(ConstantVertex.of(new double[]{1, 2, 3, 4}));

        DoubleVertex sum = prod.plus(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        DoubleVertex prod2 = sum.times(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        DoubleVertex output = prod2.plus(5).times(2);

        PartialDerivatives derivative = output.getDerivativeWrtLatents();

        DoubleTensor wrtA = derivative.withRespectTo(A);

        DoubleTensor expectedWrt = DoubleTensor.create(new double[]{4, 16, 36, 64})
            .diag()
            .reshape(TensorShape.concat(A.getShape(), A.getShape()));

        assertArrayEquals(expectedWrt.asFlatDoubleArray(), wrtA.asFlatDoubleArray(), 0.0);
        assertArrayEquals(expectedWrt.getShape(), wrtA.getShape());
    }

    @Test
    public void diffWrtScalarOverMultipleMultiplies() {

        DoubleVertex A = new UniformVertex(0, 1);
        A.setValue(2);

        DoubleVertex prod = A.times(ConstantVertex.of(new double[]{1, 2, 3, 4}));

        DoubleVertex sum = prod.plus(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        DoubleVertex prod2 = sum.times(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        DoubleVertex output = prod2.plus(5).times(2);

        PartialDerivatives derivative = output.getDerivativeWrtLatents();

        DoubleTensor wrtA = derivative.withRespectTo(A);

        DoubleTensor expectedWrt = DoubleTensor.create(new double[]{4, 16, 36, 64}).reshape(1, 4, 1, 1);

        assertArrayEquals(expectedWrt.asFlatDoubleArray(), wrtA.asFlatDoubleArray(), 0.0);
        assertArrayEquals(expectedWrt.getShape(), wrtA.getShape());
    }

    @Test
    public void diffWrtScalarOverMultipleMultipliesAndSummation() {

        DoubleVertex A = new UniformVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = A.sum().times(ConstantVertex.of(new double[]{1, 2, 3, 4})).sum();

        PartialDerivatives derivative = B.getDerivativeWrtLatents();

        DoubleTensor wrtA = derivative.withRespectTo(A);

        //B = 1*(a00 + a01 + a10 + a11) + 2*(a00 + a01 + a10 + a11)+ 3*(a00 + a01 + a10 + a11)+ 4*(a00 + a01 + a10 + a11)
        //dBda00 = 1 + 2 + 3 + 4 = 10
        DoubleTensor expectedWrt = DoubleTensor.create(new double[]{10, 10, 10, 10}).reshape(1, 1, 2, 2);

        assertThat(wrtA, equalTo(expectedWrt));
    }

}
