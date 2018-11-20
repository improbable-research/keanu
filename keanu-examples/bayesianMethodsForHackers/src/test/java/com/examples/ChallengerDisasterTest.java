package com.examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

@Slf4j
public class ChallengerDisasterTest {

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
