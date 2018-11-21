package com.examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class ChallengerDisasterTest {

    private static final Logger log = LoggerFactory.getLogger(ChallengerDisasterTest.class);

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    @Test
    public void testWhenChallengerDisasterIsRunThenBetaIsSmallAndPositive() {
        ChallengerDisaster.ChallengerPosteriors posteriors = ChallengerDisaster.run();

        log.info("mapAlpha " + posteriors.mapAlpha);
        log.info("mapBeta " + posteriors.mapBeta);

        assertThat(posteriors.mapBeta).isCloseTo(0.27, within(0.05));
        assertThat(posteriors.mapAlpha).isCloseTo(-18d, within(3d));
    }
}
