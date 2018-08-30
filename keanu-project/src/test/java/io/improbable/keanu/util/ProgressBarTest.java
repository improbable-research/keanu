package io.improbable.keanu.util;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import static junit.framework.TestCase.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

public class ProgressBarTest {

    @Test
    public void doesPrintToStream() throws UnsupportedEncodingException {

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(baos, true, "UTF-8");
        ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);

        final AtomicReference<Runnable> progressUpdateCall = new AtomicReference<>(null);

        when(scheduler.scheduleAtFixedRate(any(), anyLong(), anyLong(), any()))
            .thenAnswer(invocation -> {
                progressUpdateCall.set(invocation.getArgument(0));
                return null;
            });

        ProgressBar progressBar = new ProgressBar(printStream, scheduler);

        progressBar.progress();
        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        progressBar.progress();
        progressUpdateCall.get().run();
        progressBar.finished();

        byte[] outputBytes = baos.toByteArray();
        convertCrToNewLine(outputBytes);

        String result = new String(outputBytes, StandardCharsets.UTF_8);

        assertEquals(3, result.split("\n").length);
    }

    private void convertCrToNewLine(byte[] outputBytes) {
        for (int i = 0; i < outputBytes.length; i++) {
            if (outputBytes[i] == '\r') {
                outputBytes[i] = '\n';
            }
        }
    }

}
