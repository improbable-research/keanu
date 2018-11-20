package com.examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

@Slf4j
public class TextMessagingTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
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
