package io.improbable.keanu.util.status;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class KeanuAnimationComponentTest {

    @Test
    public void animationChangesEachRender() {
        KeanuAnimationComponent animationComponent = new KeanuAnimationComponent();
        assertThat(animationComponent.render(), equalTo("\r|Keanu|"));
        assertThat(animationComponent.render(), equalTo("\r\\Keanu/"));
    }
}
