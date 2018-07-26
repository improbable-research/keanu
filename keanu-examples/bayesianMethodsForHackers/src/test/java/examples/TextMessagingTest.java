package examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TextMessagingTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    @Test
    public void testWhenTextMessagingScenarioIsRunThenSwitchPointIsAccurate() {
        // act
        TextMessaging.TextMessagingResults output = TextMessaging.run();

        // assert
        assertThat(output.switchPointMode).isBetween(41, 46);
    }
}
