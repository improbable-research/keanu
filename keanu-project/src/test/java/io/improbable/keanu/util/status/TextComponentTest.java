package io.improbable.keanu.util.status;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class TextComponentTest {

    @Test
    public void setTextRendersCorrectly() {
        TextComponent textComponent = new TextComponent();
        String testText = "Loading";
        textComponent.setText(testText);
        assertThat(textComponent.render(), equalTo(testText));

    }
}
