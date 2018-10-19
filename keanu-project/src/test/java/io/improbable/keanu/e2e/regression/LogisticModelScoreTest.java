package io.improbable.keanu.e2e.regression;


import io.improbable.keanu.model.ModelScoring;
import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

public class LogisticModelScoreTest {
    @Test
    public void accuracyOfEverythingCorrectIs1() {
        BooleanTensor test = BooleanTensor.create(true, false, true, false);
        assertThat(ModelScoring.accuracy(test, test), closeTo(1, 1e-10));
    }

    @Test
    public void accuracyOfEverythingWrongIs0() {
        BooleanTensor test = BooleanTensor.create(true, false, true, false);
        assertThat(ModelScoring.accuracy(test, test.not()), closeTo(0, 1e-10));
    }

    @Test
    public void accuracyOfHalfIsHalf() {
        assertThat(ModelScoring.accuracy(BooleanTensor.create(true, true, true, true), BooleanTensor.create(true, false, true, false)), closeTo(0.5, 1e-10));
    }
}
