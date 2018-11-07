package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class KeanuRandomTest {

    @Test
    public void canSampleGuassisnScalar(){
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor scalar = random.nextGaussian(new long[0]);
        assertEquals(0, scalar.getRank());
    }

    @Test
    public void canSampleUniformScalar(){
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor scalar = random.nextDouble(new long[0]);
        assertEquals(0, scalar.getRank());
    }

    @Test
    public void canSampleGammaScalar(){
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor scalar = random.nextGamma(new long[0], DoubleTensor.scalar(2), DoubleTensor.scalar(2));
        assertEquals(0, scalar.getRank());
    }

    @Test
    public void canSampleLaplaceScalar(){
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor scalar = random.nextLaplace(new long[0], DoubleTensor.scalar(2), DoubleTensor.scalar(2));
        assertEquals(0, scalar.getRank());
    }
}
