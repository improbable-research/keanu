package io.improbable.keanu.e2e.regression;


import static org.junit.Assert.assertEquals;

import io.improbable.keanu.model.ModelScoring;
import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;

public class LogisticModelScoreTest {
    @Test
    public void accuracyOfEverythingCorrectIs1() {
        BooleanTensor test = BooleanTensor.create(true, false, true, false);
        assertEquals(ModelScoring.accuracy(test, test), 1, 1e-10);
    }

    @Test
    public void accuracyOfEverythingWrongIs0() {
        BooleanTensor test = BooleanTensor.create(true, false, true, false);
        assertEquals(ModelScoring.accuracy(test, test.not()), 0, 1e-10);
    }

    @Test
    public void accuracyOfHalfIsHalf() {
        BooleanTensor test = BooleanTensor.create(true, false, true, false);
        assertEquals(ModelScoring.accuracy(BooleanTensor.create(true, true, true, true), BooleanTensor.create(true, false, true, false)), 0.5, 1e-10);
    }
}
