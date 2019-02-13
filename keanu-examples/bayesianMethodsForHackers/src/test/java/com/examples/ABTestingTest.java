package com.examples;

import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ABTestingTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
        VertexId.resetIdGenerator();
    }

    @Test
    public void testWhenABTestIsRunThenBothTreatmentsHaveAccuratePosteriors() {

        ABTesting.ABTestingMaximumAPosteriori posteriors = ABTesting.run();

        assertThat(posteriors.pA).isBetween(0.02, 0.07);
        assertThat(posteriors.pB).isBetween(0.02, 0.07);
    }
}
