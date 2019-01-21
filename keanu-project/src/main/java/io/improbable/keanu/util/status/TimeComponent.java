package io.improbable.keanu.util.status;

import org.apache.commons.lang3.time.DurationFormatUtils;

import java.time.Duration;

public abstract class TimeComponent implements StatusBarComponent {

    protected String formatDuration(Duration duration) {
        return DurationFormatUtils.formatDuration(duration.toMillis(), "H:mm:ss", true);
    }
}
