package io.improbable.keanu.util.status;


import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class StatusBar {
    private final AtomicReference<String> latestMessage = new AtomicReference<>();

    private static final AtomicBoolean ENABLED = new AtomicBoolean(true);
    private static final long FRAME_PERIOD_MS = 500;
    private final AtomicInteger nextFrameIndex = new AtomicInteger(0);
    private int previouslyPrintedUpdateLength = 0;
    private final TextComponent textComponent = new TextComponent();

    private List<StatusBarComponent> components = new ArrayList<>();
    private static PrintStream defaultPrintStream = System.out;
    private static final String MIDDLE_MESSAGE = "Keanu";
    private static final String[] FRAMES = new String[]{
        "|" + MIDDLE_MESSAGE + "|",
        "\\" + MIDDLE_MESSAGE + "/",
        "-" + MIDDLE_MESSAGE + "-",
        "/" + MIDDLE_MESSAGE + "\\"
    };

    /**
     * Override the default print stream globally
     *
     * @param printStream The new printStream object to use for all StatusBars that don't declare one
     */
    public static void setDefaultPrintStream(PrintStream printStream) {
        StatusBar.defaultPrintStream = printStream;
    }

    /**
     * Disables all status bars globally
     */
    public static void disable() {
        ENABLED.set(false);
    }

    /**
     * Enables all status bars globally
     */
    public static void enable() {
        ENABLED.set(true);
    }

    protected static ScheduledExecutorService getDefaultScheduledExecutorService() {
        return Executors.newScheduledThreadPool(1, r -> {
            Thread t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        });
    }

    private final PrintStream printStream;
    private final ScheduledExecutorService scheduler;

    private final List<Runnable> onFinish = new ArrayList<>();

    public StatusBar(PrintStream printStream, ScheduledExecutorService scheduler) {
        this.printStream = printStream;
        this.scheduler = scheduler;
        // Progress bar is disabled for testing.
        String disableProgressBar = System.getProperty("io.improbable.keanu.util.ProgressBar.disableProgressBar");
        if (disableProgressBar != null && disableProgressBar.equals("true")) {
            StatusBar.disable();
        }
        addComponent(new KeanuAnimationComponent());
        addComponent(textComponent);
        startUpdateThread();
    }

    public StatusBar(PrintStream printStream) {
        this(printStream, getDefaultScheduledExecutorService());
    }

    public StatusBar(ScheduledExecutorService scheduler) {
        this(defaultPrintStream, scheduler);
    }

    public StatusBar() {
        this(defaultPrintStream, getDefaultScheduledExecutorService());
    }

    private void startUpdateThread() {
        scheduler.scheduleAtFixedRate(this::printUpdate, 0, FRAME_PERIOD_MS, TimeUnit.MILLISECONDS);
    }

    private void printUpdate() {
        if (!shouldUpdate()) {
            return;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(formatComponents());
        appendSpacesToClearPreviousContent(sb);
        printStream.print(sb.toString());
    }

    private boolean shouldUpdate() {
        return ENABLED.get();
    }

    private void appendSpacesToClearPreviousContent(StringBuilder sb) {
        int originalStringLength = sb.length();
        for (int i = originalStringLength; i < previouslyPrintedUpdateLength; i++) {
            sb.append(" ");
        }
        previouslyPrintedUpdateLength = originalStringLength;
    }

    public void finish() {
        scheduler.shutdown();
        printUpdate();
        printFinish();
        onFinish.forEach(Runnable::run);
    }

    private void printFinish() {
        if (shouldUpdate()) {
            printStream.print("\n");
        }
    }

    public void addFinishHandler(Runnable finishHandler) {
        this.onFinish.add(finishHandler);
    }

    public void setMessage(String message) {
        textComponent.setText(message);
    }

    public void addComponent(StatusBarComponent component) {
        components.add(component);
    }

    public void removeComponent(StatusBarComponent component) {
        components.remove(component);
    }

    private String formatComponents() {
        StringBuilder sb = new StringBuilder();
        for (StatusBarComponent component : components) {
            sb.append(component.render()).append(" ");
        }
        return sb.toString();
    }


}
