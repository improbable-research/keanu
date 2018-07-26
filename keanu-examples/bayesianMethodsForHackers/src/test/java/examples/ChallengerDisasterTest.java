package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ChallengerDisasterTest {
    @Test
    public void testWhenChallengerDisasterIsRunThenBetaIsSmallAndPositive() {
        // act
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        // assert
        assertThat(posteriors.mapBeta).isBetween(0.1, 0.4);
        assertThat(posteriors.mapAlpha).isBetween(-25.0, -5.0);
    }
}
