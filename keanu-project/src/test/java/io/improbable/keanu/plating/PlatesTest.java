package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

import org.junit.Test;

public class PlatesTest {
    @Test
    public void youCanGetTheLastPlate() {
        int numPlates = 10;
        Plates plates = new Plates(numPlates);
        for (int i = 0; i < numPlates-1; i++) {
            plates.add(mock(Plate.class));
        }

        Plate lastPlate = mock(Plate.class);
        plates.add(lastPlate);

        assertThat(plates.getLastPlate(), sameInstance(lastPlate));
    }

    @Test(expected = PlateException.class)
    public void itThrowsIfYouAskForTheLastPlateButThereIsNone() {
        new Plates(10).getLastPlate();
    }
}
