package io.improbable.keanu.util.status;

import java.time.Duration;

public class RemainingTimeComponent extends TimeComponent {
    private final AverageTimeComponent averageTime = new AverageTimeComponent();
    private final long totalSteps;

    public RemainingTimeComponent(long totalSteps) {
        this.totalSteps = totalSteps;
    }

    @Override
    public String render() {
        String result = averageTime.render();
        long remainingSteps = totalSteps - averageTime.getCurrentStep().get();
        Duration timeRemaining = averageTime.getAverageStepTime().multipliedBy(remainingSteps);
        result += ", Time remaining: " + formatDuration(timeRemaining);
        return result;
    }

    public void step() {
        averageTime.step();
    }
}
