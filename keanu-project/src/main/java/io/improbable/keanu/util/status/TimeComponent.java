package io.improbable.keanu.util.status;

import java.time.Duration;

public abstract class TimeComponent implements StatusBarComponent {

    protected String formatDuration(Duration duration) {
        // Duration.toString() uses the ISO-8601. This strips the `PT` at the start off.
        return duration.toString().substring(2);
    }
}
