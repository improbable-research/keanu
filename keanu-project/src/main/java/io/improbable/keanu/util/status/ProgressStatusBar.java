package io.improbable.keanu.util.status;

public class ProgressStatusBar {
    private final StatusBar statusBar;

    public ProgressStatusBar(StatusBar statusBar) {
        this.statusBar = statusBar;
    }

    public void progress(String message, double progressPercentage) {
        StringBuilder sb = new StringBuilder();
        sb.append(message);
        sb.append(" ");
        sb.append(formatProgress(progressPercentage));
        statusBar.setMessage(sb.toString());
    }

    public String formatProgress(double progressPercentage) {
        return String.format(" %3.1f%%", Math.min(100.0, Math.max(0, progressPercentage * 100)));
    }
}
