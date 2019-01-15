package io.improbable.keanu.util.status;

import lombok.Getter;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicLong;

public class AverageTimeComponent implements StatusBarComponent {
    private final ElapsedTimeComponent elapsedTime = new ElapsedTimeComponent();
    @Getter
    private AtomicLong currentStep = new AtomicLong(0);

    @Getter
    private Duration averageStepTime;

    @Override
    public String render() {
        String result = elapsedTime.render();
        if (currentStep.get() != 0) {
            averageStepTime = Duration.between(elapsedTime.getStartTime(), Instant.now()).dividedBy(currentStep.get());
            result += ", Average step time: " + averageStepTime.toString().substring(2);
        }
        return result;
    }

    public void step() {
        currentStep.getAndIncrement();
    }
}
