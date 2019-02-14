package io.improbable.keanu.templating;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

public class SequenceTest {
    @Test
    public void youCanGetTheLastItem() {
        int numItems = 10;
        Sequence sequence = new Sequence(numItems);
        for (int i = 0; i < numItems-1; i++) {
            sequence.add(mock(SequenceItem.class));
        }

        SequenceItem lastItem = mock(SequenceItem.class);
        sequence.add(lastItem);

        assertThat(sequence.getLastItem(), sameInstance(lastItem));
    }

    @Test(expected = SequenceConstructionException.class)
    public void itThrowsIfYouAskForTheLastItemButThereIsNone() {
        new Sequence(10).getLastItem();
    }
}
