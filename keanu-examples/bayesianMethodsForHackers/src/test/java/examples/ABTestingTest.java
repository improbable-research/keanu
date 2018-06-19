package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ABTestingTest {

    @Test
    public void testWhenABTestIsRunThenBothTreatmentsHaveAccuratePosteriors() {
        // act
        ABTesting.ABTestingPosteriors posteriors = ABTesting.run();

        // assert
        assertThat(posteriors.getpAMode()).isBetween(0.02, 0.07);
        assertThat(posteriors.getpBMode()).isBetween(0.02, 0.07);
    }
}
