package io.improbable.keanu.vertices.dbltensor;

import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public class KeanuRandomTest {

    Nd4jDoubleTensor matrixA;
    Nd4jDoubleTensor matrixB;
    Nd4jDoubleTensor matrixC;
    Nd4jDoubleTensor scalarA;
    Random random;

    @Before
    public void setup() {
        matrixA = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 4});
        matrixB = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 4});
        matrixC = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 4});
        scalarA = Nd4jDoubleTensor.scalar(2.0);
        random = new Random(1);
    }

    @Test
    public void canSampleFromGamma() {
        KeanuRandom keanuRandom = new KeanuRandom();
        DoubleTensor gamma = keanuRandom.nextGamma(matrixA.getShape(), matrixA, matrixB, matrixC, random);
    }

}
