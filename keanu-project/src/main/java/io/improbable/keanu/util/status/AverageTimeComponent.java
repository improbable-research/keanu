package io.improbable.keanu.util.status;

import lombok.Getter;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link StatusBarComponent} that renders the average time algorithm steps are taking
 * in the form of steps per second, also shows elapsed time.
 */
public class AverageTimeComponent extends TimeComponent {

    private final ElapsedTimeComponent elapsedTime = new ElapsedTimeComponent();
    private static final long NANOS_IN_SECOND = 1000000000;

    @Getter
    private AtomicLong currentStep = new AtomicLong(0);

    @Getter
    private Duration averageStepTime;

    @Override
    public String render() {
        StringBuilder renderedString = new StringBuilder(elapsedTime.render());
        long currentStepNow = currentStep.get();

        if (currentStepNow != 0) {
            averageStepTime = Duration.between(elapsedTime.getStartTime(), Instant.now()).dividedBy(currentStepNow);
            long averageStepTimeNanos = averageStepTime.toNanos();
            if(averageStepTimeNanos != 0) {
                double stepsSecond = (double) NANOS_IN_SECOND / averageStepTime.toNanos();
                renderedString.append(", Steps per second: ");
                renderedString.append(String.format("%.3f", stepsSecond));
            }
        }

        return renderedString.toString();
    }

    /**
     * Advances the step counter which is used to calculate the average step time.
     */
    public void step() {
        currentStep.getAndIncrement();
    }
}
