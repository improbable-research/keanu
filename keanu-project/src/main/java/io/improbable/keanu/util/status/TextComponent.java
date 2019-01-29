package io.improbable.keanu.util.status;

import java.util.concurrent.atomic.AtomicReference;

public class TextComponent implements StatusBarComponent {

    private AtomicReference<String> content = new AtomicReference<>("");

    /**
     * Sets the text to be displayed by the component.
     * @param text the text to display.
     */
    public void setText(String text) {
        content.set(text);
    }

    @Override
    public String render() {
        return content.get();
    }
}
