package io.improbable.keanu.util;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class ProgressBar {

    private static PrintStream defaultPrintStream = System.out;

    /**
     * Override the default print stream globally
     *
     * @param printStream The new printStream object to use for all ProgressBars that don't declare one
     */
    public static void setDefaultPrintStream(PrintStream printStream) {
        ProgressBar.defaultPrintStream = printStream;
    }

    /**
     * Disables all progress bars globally
     */
    public static void disable() {
        ENABLED.set(false);
    }

    /**
     * Enables all progress bars globally
     */
    public static void enable() {
        ENABLED.set(true);
    }

    private static ScheduledExecutorService getDefaultScheduledExecutorService() {
        return Executors.newScheduledThreadPool(1, r -> {
            Thread t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        });
    }

    private static final AtomicBoolean ENABLED = new AtomicBoolean(true);
    private static final long FRAME_PERIOD_MS = 500;
    private static final ProgressUpdate DEFAULT_UPDATE = new ProgressUpdate();
    private static final String MIDDLE_MESSAGE = "Keanu";
    private static final String[] FRAMES = new String[]{
        "|" + MIDDLE_MESSAGE + "|",
        "\\" + MIDDLE_MESSAGE + "/",
        "-" + MIDDLE_MESSAGE + "-",
        "/" + MIDDLE_MESSAGE + "\\"
    };

    private final ScheduledExecutorService scheduler;
    private final PrintStream printStream;

    private final AtomicReference<ProgressUpdate> latestProgressUpdate = new AtomicReference<>();
    private final AtomicInteger nextFrameIndex = new AtomicInteger(0);

    private final List<Runnable> onFinish = new ArrayList<>();

    public ProgressBar(PrintStream printStream, ScheduledExecutorService scheduler) {
        this.printStream = printStream;
        this.scheduler = scheduler;
        startUpdateThread();
    }

    public ProgressBar(PrintStream printStream) {
        this(printStream, getDefaultScheduledExecutorService());
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

    private boolean shouldUpdate() {
        return ENABLED.get();
    }

    public ProgressUpdate getProgress() {
        return latestProgressUpdate.get();
    }

    public void finish() {
        scheduler.shutdown();
        printUpdate();
        printFinish();
        onFinish.forEach(Runnable::run);
    }

    public void addFinishHandler(Runnable finishHandler) {
        this.onFinish.add(finishHandler);
    }

    private void startUpdateThread() {
        scheduler.scheduleAtFixedRate(this::printUpdate, 0, FRAME_PERIOD_MS, TimeUnit.MILLISECONDS);
    }

    private void printUpdate() {

        if (!ENABLED.get()) {
            return;
        }

        ProgressUpdate update = latestProgressUpdate.get();

        if (update != null) {

            StringBuilder sb = new StringBuilder();

            sb.append("\r").append(FRAMES[nextFrameIndex.getAndIncrement() % FRAMES.length]);

            if (update.getMessage() != null) {
                sb.append(" ");
                sb.append(update.getMessage());
                sb.append(" ");
            }

            if (update.getProgressPercentage() != null) {
                sb.append(String.format(" %3.1f%%", Math.min(100.0, Math.max(0, update.getProgressPercentage() * 100))));
            }

            printStream.print(sb.toString());
        }
    }

    private void printFinish() {
        if (ENABLED.get()) {
            printStream.print("\n");
        }
    }
}
