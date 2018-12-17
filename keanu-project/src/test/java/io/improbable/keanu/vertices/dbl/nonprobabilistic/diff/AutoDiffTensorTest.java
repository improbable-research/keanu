package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;

public class AutoDiffTensorTest {

    @Test
    public void diffWrtVectorOverMultipleMultiplies() {

        DoubleVertex A = new UniformVertex(new long[]{1, 4}, 0, 1);
        A.setValue(new double[]{-1, 3, 5, -2});

        DoubleVertex prod = A.times(ConstantVertex.of(new double[]{1, 2, 3, 4}));

        DoubleVertex sum = prod.plus(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        DoubleVertex prod2 = sum.times(ConstantVertex.of(new double[]{2, 4, 6, 8}));

        MultiplicationVertex output = prod2.plus(5).times(2);

        DoubleTensor wrtA = Differentiator.reverseModeAutoDiff(output, A).withRespectTo(A).getPartial();

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

        MultiplicationVertex output = prod2.plus(5).times(2);

        DoubleTensor wrtA = Differentiator.reverseModeAutoDiff(output, A).withRespectTo(A).getPartial();

        DoubleTensor expectedWrt = DoubleTensor.create(4, 16, 36, 64);

        assertArrayEquals(expectedWrt.asFlatDoubleArray(), wrtA.asFlatDoubleArray(), 0.0);
        assertArrayEquals(expectedWrt.getShape(), wrtA.getShape());
    }

    @Test
    public void diffWrtScalarOverMultipleMultipliesAndSummation() {

        DoubleVertex A = new UniformVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        SumVertex B = A.sum().times(ConstantVertex.of(new double[]{1, 2, 3, 4})).sum();

        DoubleTensor wrtA = Differentiator.reverseModeAutoDiff(B, A).withRespectTo(A).getPartial();

        //B = 1*(a00 + a01 + a10 + a11) + 2*(a00 + a01 + a10 + a11)+ 3*(a00 + a01 + a10 + a11)+ 4*(a00 + a01 + a10 + a11)
        //dBda00 = 1 + 2 + 3 + 4 = 10
        DoubleTensor expectedWrt = DoubleTensor.create(new double[]{10, 10, 10, 10}).reshape(2, 2);

        assertThat(wrtA, equalTo(expectedWrt));
    }

}
