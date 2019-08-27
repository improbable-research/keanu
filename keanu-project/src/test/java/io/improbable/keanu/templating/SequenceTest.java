package io.improbable.keanu.templating;

import io.improbable.keanu.network.BayesianNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

public class SequenceTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanGetTheLastItem() {
        int numItems = 10;
        Sequence sequence = new Sequence(numItems, this.hashCode(), null);
        for (int i = 0; i < numItems - 1; i++) {
            sequence.add(mock(SequenceItem.class));
        }

        SequenceItem lastItem = mock(SequenceItem.class);
        sequence.add(lastItem);

        assertThat(sequence.getLastItem(), sameInstance(lastItem));
    }

    @Test(expected = SequenceConstructionException.class)
    public void itThrowsIfYouAskForTheLastItemButThereIsNone() {
        new Sequence(10, this.hashCode(), null).getLastItem();
    }

    @Test
    public void bayesNetConstructionFailsWhenThereAreNoSequenceItems() {
        expectedException.expect(RuntimeException.class);
        expectedException.expectMessage("Bayesian Network construction failed because the Sequence contains no SequenceItems");
        Sequence sequence = new Sequence(0, this.hashCode(), null);
        BayesianNetwork network = sequence.toBayesianNetwork();
    }

    @Test
    public void bayesNetConstructionFailsWhenThereAreNoVertices() {
        expectedException.expect(RuntimeException.class);
        expectedException.expectMessage("Bayesian Network construction failed because there are no vertices in the Sequence");
        SequenceItem emptySequenceItem = new SequenceItem(0, 0);
        Sequence sequence = new Sequence(0, this.hashCode(), null);
        sequence.add(emptySequenceItem);
        BayesianNetwork network = sequence.toBayesianNetwork();
    }
}
