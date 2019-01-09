package io.improbable.keanu.util;

import io.improbable.keanu.util.status.StatusBar;

import java.io.PrintStream;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

public class ProgressBar extends StatusBar {
    private static final ProgressUpdate DEFAULT_UPDATE = new ProgressUpdate();
    private final AtomicReference<ProgressUpdate> latestProgressUpdate = new AtomicReference<>();

    public ProgressBar(PrintStream printStream, ScheduledExecutorService scheduler) {
        super(printStream, scheduler);

    }

    public ProgressBar(PrintStream printStream) {
        this(printStream, getDefaultScheduledExecutorService());
    }

    public ProgressBar(ScheduledExecutorService scheduler) {
        this(defaultPrintStream, scheduler);
    }

    public ProgressBar() {
        this(defaultPrintStream, getDefaultScheduledExecutorService());
    }

    public void progress() {
        progress(DEFAULT_UPDATE);
    }

    public void progress(String message, Double progressPercentage) {
        if (shouldUpdate()) {
            progress(new ProgressUpdate(message, progressPercentage));
        }
    }

    public void progress(String message) {
        if (shouldUpdate()) {
            progress(new ProgressUpdate(message));
        }
    }

    public void progress(Double progressPercentage) {
        if (shouldUpdate()) {
            progress(new ProgressUpdate(progressPercentage));
        }
    }

    public void progress(ProgressUpdate progressUpdate) {
        latestProgressUpdate.set(progressUpdate);
    }

    public ProgressUpdate getProgress() {
        return latestProgressUpdate.get();
    }

    @Override
    protected String formatContent() {
        StringBuilder sb = new StringBuilder();
        ProgressUpdate update = latestProgressUpdate.get();
        if (update != null) {
            if (update.getMessage() != null) {
                sb.append(" ");
                sb.append(update.getMessage());
                sb.append(" ");
            }
            if (update.getProgressPercentage() != null) {
                sb.append(String.format(" %3.1f%%", Math.min(100.0, Math.max(0, update.getProgressPercentage() * 100))));
            }
        }
        return sb.toString();
    }
}
