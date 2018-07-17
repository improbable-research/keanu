package io.improbable.keanu.dual;

import static org.junit.Assert.assertTrue;

import static io.improbable.keanu.distributions.dual.Duals.MU;

import java.util.NoSuchElementException;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.Dual;
import io.improbable.keanu.distributions.dual.Duals;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class DualTest {
    Duals duals;

    @Before
    public void initialiseDuals() {
        duals = new Duals();
    }

    @Test
    public void youCanGetADualByName() {
        DoubleTensor muDualValue = DoubleTensor.scalar(0.1);
        duals.put(MU, muDualValue);
        Dual mu = duals.get(MU);
        assertTrue(mu.getName().equals(MU.getName()));
        assertTrue(mu.getValue() == muDualValue);
    }

    @Test(expected = NoSuchElementException.class)
    public void itThrowsIfYouAskForAValueThatsAbsent() {
        duals.get(MU);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTheSameDualTwice() {
        DoubleTensor muDualValue = DoubleTensor.scalar(0.1);
        duals.put(MU, muDualValue);
        duals.put(MU, muDualValue);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTwoDualsWithTheSameName() {
        duals.put(MU, DoubleTensor.scalar(0.1));
        duals.put(MU, DoubleTensor.scalar(0.2));
    }
}
