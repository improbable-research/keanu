package io.improbable.keanu.util.status;

public class ProgressBar {
    private final StatusBar statusBar;

    public ProgressBar(StatusBar statusBar) {
        this.statusBar = statusBar;
    }

    public void progress(String message, double progressPercentage) {
        StringBuilder sb = new StringBuilder();
        if(message != null) {
            sb.append(message);
            sb.append(" ");
        }

        sb.append(formatProgress(progressPercentage));
        statusBar.setMessage(sb.toString());
    }

    public void progress(double progressPercentage) {
        progress(null, progressPercentage);
    }

    public String formatProgress(double progressPercentage) {
        return String.format(" %3.1f%%", Math.min(100.0, Math.max(0, progressPercentage * 100)));
    }
}
