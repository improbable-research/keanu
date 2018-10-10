package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static org.junit.Assert.assertTrue;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;

import java.util.NoSuchElementException;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.hyperparam.Diff;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class DiffTest {
    Diffs diffs;

    @Before
    public void initialiseDiffs() {
        diffs = new Diffs();
    }

    @Test
    public void youCanGetADiffByName() {
        DoubleTensor muDiffValue = DoubleTensor.scalar(0.1);
        diffs.put(MU, muDiffValue);
        Diff mu = diffs.get(MU);
        assertTrue(mu.getName().equals(MU.getName()));
        assertTrue(mu.getValue() == muDiffValue);
    }

    @Test(expected = NoSuchElementException.class)
    public void itThrowsIfYouAskForAValueThatsAbsent() {
        diffs.get(MU);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTheSameDiffTwice() {
        DoubleTensor muDiffValue = DoubleTensor.scalar(0.1);
        diffs.put(MU, muDiffValue);
        diffs.put(MU, muDiffValue);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfYouTryToAddTwoDiffsWithTheSameName() {
        diffs.put(MU, DoubleTensor.scalar(0.1));
        diffs.put(MU, DoubleTensor.scalar(0.2));
    }
}
