package io.improbable.keanu.util.status;

import java.util.concurrent.atomic.AtomicInteger;

public class KeanuAnimationComponent implements StatusBarComponent {
    private final AtomicInteger nextFrameIndex = new AtomicInteger(0);
    private static final String MIDDLE_MESSAGE = "Keanu";
    private static final String[] FRAMES = new String[]{
        "|" + MIDDLE_MESSAGE + "|",
        "\\" + MIDDLE_MESSAGE + "/",
        "-" + MIDDLE_MESSAGE + "-",
        "/" + MIDDLE_MESSAGE + "\\"
    };

    @Override
    public String render() {
        String result = "\r";
        result += FRAMES[nextFrameIndex.getAndIncrement() % FRAMES.length];
        return result;
    }
}
