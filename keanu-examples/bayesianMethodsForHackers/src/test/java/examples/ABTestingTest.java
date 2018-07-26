package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ABTestingTest {

    @Test
    public void testWhenABTestIsRunThenBothTreatmentsHaveAccuratePosteriors() {
        // act
        ABTesting.ABTestingMaximumAPosteriori posteriors = ABTesting.run();

        // assert
        assertThat(posteriors.pA).isBetween(0.02, 0.07);
        assertThat(posteriors.pB).isBetween(0.02, 0.07);
    }
}
