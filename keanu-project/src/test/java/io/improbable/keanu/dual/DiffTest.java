package io.improbable.keanu.dual;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static org.junit.Assert.assertTrue;

import io.improbable.keanu.distributions.dual.Diff;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import java.util.NoSuchElementException;
import org.junit.Before;
import org.junit.Test;

public class DiffTest {
  Diffs diffs;

  @Before
  public void initialiseDuals() {
    diffs = new Diffs();
  }

  @Test
  public void youCanGetADualByName() {
    DoubleTensor muDualValue = DoubleTensor.scalar(0.1);
    diffs.put(MU, muDualValue);
    Diff mu = diffs.get(MU);
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
