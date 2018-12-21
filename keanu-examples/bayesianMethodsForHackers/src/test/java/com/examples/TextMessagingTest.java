package com.examples;

import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TextMessagingTest {

    private static final Logger log = LoggerFactory.getLogger(TextMessagingTest.class);

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
        VertexId.ID_GENERATOR.set(1);
    }

    @Test
    public void testWhenTextMessagingScenarioIsRunThenSwitchPointIsAccurate() {
        TextMessaging.TextMessagingResults output = TextMessaging.run();

        log.info("Switch Point Mode " + output.switchPointMode);
        log.info("Early Rate Mean " + output.earlyRateMean);
        log.info("Late Rate Mean " + output.lateRateMean);

        assertThat(output.switchPointMode).isCloseTo(43, within(2));
        assertThat(output.earlyRateMean).isCloseTo(18, within(2d));
        assertThat(output.lateRateMean).isCloseTo(23, within(2d));
    }
}
