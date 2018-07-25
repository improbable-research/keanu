package io.improbable.keanu.dual;

import static org.junit.Assert.assertTrue;

import static io.improbable.keanu.distributions.dual.ParameterName.MU;

import java.util.NoSuchElementException;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterValue;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class DiffTest {
    ParameterMap<DoubleTensor> diffs;

    @Before
    public void initialiseDuals() {
        diffs = new ParameterMap<DoubleTensor>();
    }

    @Test
    public void youCanGetADualByName() {
        DoubleTensor muDualValue = DoubleTensor.scalar(0.1);
        diffs.put(MU, muDualValue);
        ParameterValue<DoubleTensor> mu = diffs.get(MU);
        assertTrue(mu.getName().equals(MU.getName()));
        assertTrue(mu.getValue() == muDualValue);
    }

    @Test(expected = NoSuchElementException.class)
    public void itThrowsIfYouAskForAValueThatsAbsent() {
        diffs.get(MU);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTheSameDualTwice() {
        DoubleTensor muDualValue = DoubleTensor.scalar(0.1);
        diffs.put(MU, muDualValue);
        diffs.put(MU, muDualValue);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTwoDualsWithTheSameName() {
        diffs.put(MU, DoubleTensor.scalar(0.1));
        diffs.put(MU, DoubleTensor.scalar(0.2));
    }
}
