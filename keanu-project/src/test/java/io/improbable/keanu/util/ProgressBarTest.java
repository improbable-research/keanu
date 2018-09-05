package io.improbable.keanu.util;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class ProgressBarTest {

    private AtomicReference<Runnable> progressUpdateCall;
    private ProgressBar progressBar;
    private ByteArrayOutputStream byteArrayOutputStream;

    @Before
    public void setup() throws UnsupportedEncodingException {

        byteArrayOutputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(byteArrayOutputStream, true, "UTF-8");
        ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);

        progressUpdateCall = new AtomicReference<>(null);

        when(scheduler.scheduleAtFixedRate(any(), anyLong(), anyLong(), any()))
            .thenAnswer(invocation -> {
                progressUpdateCall.set(invocation.getArgument(0));
                return null;
            });

        progressBar = new ProgressBar(printStream, scheduler);
    }

    @After
    public void cleanup() throws IOException {
        byteArrayOutputStream.close();
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

    @Test
    public void doesPrintToStreamWhenEnabled() {
        ProgressBar.enable();

        progressBar.progress();
        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        progressBar.progress();
        progressUpdateCall.get().run();
        progressBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertEquals(3, result.split("\n").length);
    }

    @Test
    public void doesNotPrintToStreamWhenGloballyDisabled() {
        ProgressBar.disable();

        progressBar.progress();
        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        progressBar.progress();
        progressBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertEquals("", result);
    }

}
