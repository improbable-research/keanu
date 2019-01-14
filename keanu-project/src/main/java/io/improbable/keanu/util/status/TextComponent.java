package io.improbable.keanu.util.status;

import java.util.concurrent.atomic.AtomicReference;

public class TextComponent implements StatusBarComponent {
    private AtomicReference<String> content = new AtomicReference<>();

    public void setText(String text) {
        content.set(text);
    }

    @Override
    public String render() {
        return content.get();
    }
}
