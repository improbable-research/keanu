package io.improbable.keanu.util.status;

import lombok.Getter;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicLong;

public class AverageTimeComponent extends TimeComponent {
    private final ElapsedTimeComponent elapsedTime = new ElapsedTimeComponent();

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
            renderedString.append(", Average step time: ");
            renderedString.append(formatDuration(averageStepTime));
        }

        return renderedString.toString();
    }

    public void step() {
        currentStep.getAndIncrement();
    }
}
