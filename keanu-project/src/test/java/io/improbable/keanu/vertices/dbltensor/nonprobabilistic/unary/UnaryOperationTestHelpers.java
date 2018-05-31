package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;

import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class UnaryOperationTestHelpers {

    public static void operatesOnScalarVertexValue(double aValue,
                                                   double expected,
                                                   Function<DoubleTensorVertex, DoubleTensorVertex> op) {

        TensorUniformVertex A = new TensorUniformVertex(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));

        assertEquals(expected, op.apply(A).getValue().scalar(), 1e-5);
    }

    public static void calculatesDualNumberOfScalar(double aValue,
                                                    double expectedGradientWrtA,
                                                    Function<DoubleTensorVertex, DoubleTensorVertex> op) {

        TensorUniformVertex A = new TensorUniformVertex(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));

        TensorDualNumber resultDualNumber = op.apply(A).getDualNumber();
        assertEquals(expectedGradientWrtA, resultDualNumber.getPartialDerivatives().withRespectTo(A).scalar(), 1e-5);
    }

    public static void operatesOn2x2MatrixVertexValues(double[] aValues,
                                                       double[] expected,
                                                       Function<DoubleTensorVertex, DoubleTensorVertex> op) {

        TensorUniformVertex A = new TensorUniformVertex(new int[]{2, 2}, new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0));
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, new int[]{2, 2}));

        DoubleTensor result = op.apply(A).getValue();

        DoubleTensor expectedTensor = Nd4jDoubleTensor.create(expected, new int[]{2, 2});

        assertEquals(expectedTensor.getValue(0, 0), result.getValue(0, 0), 1e-5);
        assertEquals(expectedTensor.getValue(0, 1), result.getValue(0, 1), 1e-5);
        assertEquals(expectedTensor.getValue(1, 0), result.getValue(1, 0), 1e-5);
        assertEquals(expectedTensor.getValue(1, 1), result.getValue(1, 1), 1e-5);
    }

    public static void calculatesDualNumberOfMatrixElementWiseOperator(double[] aValues,
                                                                       double[] expectedGradientWrtA,
                                                                       Function<DoubleTensorVertex, DoubleTensorVertex> op) {
        TensorUniformVertex A = new TensorUniformVertex(new int[]{2, 2}, new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0));
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, new int[]{2, 2}));

        TensorDualNumber result = op.apply(A).getDualNumber();
        DoubleTensor expectedTensorA = Nd4jDoubleTensor.create(expectedGradientWrtA, new int[]{2, 2});

        DoubleTensor wrtA = result.getPartialDerivatives().withRespectTo(A);
        System.out.println(wrtA.getValue(0, 0));
        assertEquals(expectedTensorA.getValue(0, 0), wrtA.getValue(0, 0), 1e-5);
        assertEquals(expectedTensorA.getValue(0, 1), wrtA.getValue(0, 1), 1e-5);
        assertEquals(expectedTensorA.getValue(1, 0), wrtA.getValue(1, 0), 1e-5);
        assertEquals(expectedTensorA.getValue(1, 1), wrtA.getValue(1, 1), 1e-5);
    }
}
