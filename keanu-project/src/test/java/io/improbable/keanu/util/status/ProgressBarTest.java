package io.improbable.keanu.util.status;

import io.improbable.keanu.util.ProgressBar;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;


public class ProgressBarTest {
    private AtomicReference<Runnable> progressUpdateCall;
    private StatusBar statusBar;
    private ProgressStatusBar progressBar;
    private ByteArrayOutputStream byteArrayOutputStream;
    private ScheduledExecutorService scheduler;

    @Before
    public void setup() throws UnsupportedEncodingException {

        byteArrayOutputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(byteArrayOutputStream, true, "UTF-8");
        scheduler = mock(ScheduledExecutorService.class);

        progressUpdateCall = new AtomicReference<>(null);

        when(scheduler.scheduleAtFixedRate(any(), anyLong(), anyLong(), any()))
            .thenAnswer(invocation -> {
                progressUpdateCall.set(invocation.getArgument(0));
                return null;
            });

        statusBar = new StatusBar(printStream, scheduler);
        progressBar = new ProgressStatusBar(statusBar);
        StatusBar.enable();
    }

    @Test
    public void doesPrintProgressInAppropriateFormat() {
        ProgressBar.enable();

        progressBar.progress(0.0);
        progressUpdateCall.get().run();
        progressBar.progress(0.675);
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertThat(result, containsString("67.5%"));
    }

    @Test
    public void doesLimitProgressTo100Percent() {
        ProgressBar.enable();

        progressBar.progress(-0.7);
        progressUpdateCall.get().run();
        progressBar.progress(1.5);
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();
        String[] lines = result.split("\n");

        assertThat(lines[1], containsString("0.0%"));
        assertThat(lines[2], containsString("100.0%"));
    }

    @After
    public void tearDown() throws Exception {
        StatusBar.setDefaultPrintStream(System.out);
        StatusBar.disable();
    }

    private void convertCrToNewLine(byte[] outputBytes) {
        for (int i = 0; i < outputBytes.length; i++) {
            if (outputBytes[i] == '\r') {
                outputBytes[i] = '\n';
            }
        }
    }

    private String getResultWithNewLinesInsteadOfCR() {
        byte[] outputBytes = byteArrayOutputStream.toByteArray();
        convertCrToNewLine(outputBytes);
        return new String(outputBytes, StandardCharsets.UTF_8);
    }

}
