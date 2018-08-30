package io.improbable.keanu.util;

import java.io.PrintStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class ProgressBar {

    private static final String MIDDLE_MESSAGE = "Keanu";
    private static final String[] FRAMES = new String[]{
        "| " + MIDDLE_MESSAGE + " |",
        "\\ " + MIDDLE_MESSAGE + " /",
        "- " + MIDDLE_MESSAGE + " -",
        "/ " + MIDDLE_MESSAGE + " \\"
    };

    private static final long FRAME_PERIOD_MS = 500;

    private final ScheduledExecutorService scheduler;
    private final AtomicBoolean hasProgressed = new AtomicBoolean(true);
    private final PrintStream printStream;

    public ProgressBar() {
        this.printStream = System.out;
        this.scheduler = Executors.newScheduledThreadPool(1, r -> {
            Thread t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        });
        startUpdateThread();
    }

    public ProgressBar(PrintStream printStream, ScheduledExecutorService scheduler) {
        this.printStream = printStream;
        this.scheduler = scheduler;
        startUpdateThread();
    }

    public void progress() {
        hasProgressed.getAndSet(true);
    }

    public void finished() {
        hasProgressed.set(false);
        scheduler.shutdown();
        printStream.print("\n");
    }

    private void startUpdateThread() {

        final AtomicInteger nextFrameIndex = new AtomicInteger(0);

        scheduler.scheduleAtFixedRate(() -> {
            if (hasProgressed.getAndSet(false)) {
                printStream.print("\r" + FRAMES[nextFrameIndex.getAndIncrement() % FRAMES.length]);
            }
        }, 0, FRAME_PERIOD_MS, TimeUnit.MILLISECONDS);

    }
}
