package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

public class PowerVertexTest {

    @Test
    public void powerTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 8.0, DoubleVertex::pow);
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsPower() {
        calculatesDualNumberOfTwoScalars(2.0, 3.0, 3.*4., Math.log(2.)*8., DoubleVertex::pow);
    }

    @Test
    public void powerTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 3.0, 4.0},
            new double[]{2.0, 3.0, 4.0, 5.0},
            new double[]{1.0, 8.0, 81.0, 1024.0},
            DoubleVertex::pow
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWisePower() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            new double[]{1.0, 2.0, 3.0, 4.0},
            new double[]{2.0, 3.0, 4.0, 5.0},
            new double[]{2.0, 3.0*4, 4.0*27, 5.0*256},
            new double[]{Math.log(1.0)*1, Math.log(2.0)*8, Math.log(3.0)*81, Math.log(4.0)*1024},
            DoubleVertex::pow
        );
    }

}
