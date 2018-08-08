package examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ChallengerDisasterTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    @Test
    public void testWhenChallengerDisasterIsRunThenBetaIsSmallAndPositive() {
        // act
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        System.out.println("mapAlpha " + posteriors.mapAlpha);
        System.out.println("mapBeta " + posteriors.mapBeta);

        // assert
        assertThat(posteriors.mapBeta).isBetween(0.1, 0.4);
        assertThat(posteriors.mapAlpha).isBetween(-25.0, -5.0);
    }
}
