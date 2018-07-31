package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.function.Function;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiator;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class UnaryOperationTestHelpers {

    public static void operatesOnScalarVertexValue(double aValue,
                                                   double expected,
                                                   Function<DoubleVertex, DoubleVertex> op) {

        ConstantDoubleVertex A = ConstantVertex.of(aValue);

        assertEquals(expected, op.apply(A).getValue().scalar(), 1e-5);
    }

    public static void calculatesDualNumberOfScalar(double aValue,
                                                    double expectedGradientWrtA,
                                                    Function<DoubleVertex, DoubleVertex> op) {

        UniformVertex A = VertexOfType.uniform(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));

        DualNumber resultDualNumber = new Differentiator().calculateDual(op.apply(A));
        assertEquals(expectedGradientWrtA, resultDualNumber.getPartialDerivatives().withRespectTo(A).scalar(), 1e-5);
    }

    public static void operatesOn2x2MatrixVertexValues(double[] aValues,
                                                       double[] expected,
                                                       Function<DoubleVertex, DoubleVertex> op) {

        ConstantDoubleVertex A = new ConstantDoubleVertex(DoubleTensor.create(aValues, new int[]{2, 2}));

        DoubleTensor result = op.apply(A).getValue();

        DoubleTensor expectedTensor = Nd4jDoubleTensor.create(expected, new int[]{2, 2});

        assertEquals(expectedTensor.getValue(0, 0), result.getValue(0, 0), 1e-5);
        assertEquals(expectedTensor.getValue(0, 1), result.getValue(0, 1), 1e-5);
        assertEquals(expectedTensor.getValue(1, 0), result.getValue(1, 0), 1e-5);
        assertEquals(expectedTensor.getValue(1, 1), result.getValue(1, 1), 1e-5);
    }

    public static void calculatesDualNumberOfMatrixElementWiseOperator(double[] aValues,
                                                                       double[] expectedGradientWrtA,
                                                                       Function<DoubleVertex, DoubleVertex> op) {

        int[] matrixShape = new int[]{2, 2};
        UniformVertex A = new DistributionVertexBuilder()
            .shaped(matrixShape)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 1.)
            .uniform();
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, matrixShape));

        DualNumber result = new Differentiator().calculateDual(op.apply(A));

        DoubleTensor wrtA = result.getPartialDerivatives().withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA, wrtA.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(TensorShape.concat(matrixShape, matrixShape), wrtA.getShape());
    }
}
