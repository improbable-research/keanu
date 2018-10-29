package com.examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class ChallengerDisasterTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    @Test
    public void testWhenChallengerDisasterIsRunThenBetaIsSmallAndPositive() {
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        System.out.println("mapAlpha " + posteriors.mapAlpha);
        System.out.println("mapBeta " + posteriors.mapBeta);

        assertThat(posteriors.mapBeta).isCloseTo(0.25, within(0.15));
        assertThat(posteriors.mapAlpha).isCloseTo(-15d, within(5d));
    }
}
