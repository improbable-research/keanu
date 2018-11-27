package io.improbable.keanu.util;

import io.improbable.keanu.testcategory.Slow;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class ProgressBarTest {

    private AtomicReference<Runnable> progressUpdateCall;
    private ProgressBar progressBar;
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

        assertEquals(5, result.split("\n").length);
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

    @Test
    public void doesPrintProgressInAppropriateFormat() {
        ProgressBar.enable();

        progressBar.progress(0.0);
        progressUpdateCall.get().run();
        progressBar.progress(0.675);
        progressBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertThat(result, containsString("67.5%"));
    }

    @Test
    public void doesLimitProgressTo100Percent() {
        ProgressBar.enable();

        progressBar.progress(-0.7);
        progressUpdateCall.get().run();
        progressBar.progress(1.5);
        progressBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();
        String[] lines = result.split("\n");

        assertThat(lines[1], containsString("0.0%"));
        assertThat(lines[2], containsString("100.0%"));
    }

    @Test
    public void doesCallFinishHandler() {
        ProgressBar.enable();

        Runnable finishHandler = mock(Runnable.class);
        progressBar.addFinishHandler(finishHandler);
        progressBar.progress();
        progressUpdateCall.get().run();
        progressBar.finish();

        verify(finishHandler).run();
        verifyNoMoreInteractions(finishHandler);
    }

    @Category(Slow.class)
    @Test
    public void youCanOverrideTheDefaultPrintStream() {
        PrintStream mockStream = mock(PrintStream.class);
        doAnswer(new Answer() {
            @Override
            public Object answer(InvocationOnMock invocation) throws Throwable {
                System.out.println(invocation.getArgument(0).toString());
                return null;
            }
        }).when(mockStream).print(anyString());

        ProgressBar.setDefaultPrintStream(mockStream);
        ProgressBar.enable();
        ProgressBar progressBar = new ProgressBar(scheduler);
        progressBar.progress();
        progressBar.finish();
        verify(mockStream, atLeastOnce()).print("\r|Keanu|");
    }

    @After
    public void tearDown() throws Exception {
        ProgressBar.setDefaultPrintStream(System.out);
    }
}
