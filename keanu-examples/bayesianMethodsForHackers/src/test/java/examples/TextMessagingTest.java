package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TextMessagingTest {
    @Test
    public void testWhenTextMessagingScenarioIsRunThenSwitchPointIsAccurate() {
        // act
        TextMessaging.TextMessagingResults output = TextMessaging.run();

        // assert
        assertThat(output.getSwitchPointMode()).isBetween(41, 46);
    }
}
