package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ChallengerDisasterTest {
    @Test
    public void testWhenChallengerDisasterIsRunThenBetaIsSmallAndPositive() {
        // act
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        // assert
        assertThat(posteriors.getAlphaMode()).isBetween(0.1, 0.4);
    }

    @Test
    public void testWhenChallengerDisasterIsRunThenAlphaIsLargeAndNegative() {
        // act
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        // assert
        assertThat(posteriors.getBetaMode()).isBetween(-25.0, -5.0);
    }
}
