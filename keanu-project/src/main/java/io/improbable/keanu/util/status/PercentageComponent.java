package io.improbable.keanu.util.status;

import com.google.common.util.concurrent.AtomicDouble;

public class PercentageComponent implements StatusBarComponent {

    private AtomicDouble percentage = new AtomicDouble(0.0);

    public void progress(double percentage) {
        this.percentage.set(percentage);
    }

    @Override
    public String render() {
        return formatProgress(percentage.get());
    }

    private String formatProgress(double progressPercentage) {
        return String.format(" %3.1f%%", Math.min(100.0, Math.max(0, progressPercentage * 100)));
    }
}
