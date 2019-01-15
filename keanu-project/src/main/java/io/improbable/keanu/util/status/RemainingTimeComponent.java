package io.improbable.keanu.util.status;

import java.time.Duration;

/**
 * {@link StatusBarComponent} that shows the estimated remaining time for algorithm completion. Also shows average step
 * time and elapsed time.
 */
public class RemainingTimeComponent extends TimeComponent {

    private final AverageTimeComponent averageTime = new AverageTimeComponent();
    private final long totalSteps;

    public RemainingTimeComponent(long totalSteps) {
        this.totalSteps = totalSteps;
    }

    @Override
    public String render() {
        StringBuilder renderedString = new StringBuilder(averageTime.render());
        long remainingSteps = totalSteps - averageTime.getCurrentStep().get();
        Duration timeRemaining = averageTime.getAverageStepTime().multipliedBy(remainingSteps);

        renderedString.append(", Time remaining: ");
        renderedString.append(formatDuration(timeRemaining));
        return renderedString.toString();
    }

    /**
     * Advances the step counter which is used to calculate the average step time.
     */
    public void step() {
        averageTime.step();
    }
}
