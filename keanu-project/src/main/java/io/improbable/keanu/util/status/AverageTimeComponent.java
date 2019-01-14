package io.improbable.keanu.util.status;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.concurrent.atomic.AtomicLong;

public class AverageTimeComponent implements StatusBarComponent {
    private final ElapsedTimeComponent elapsedTime = new ElapsedTimeComponent();
    private AtomicLong currentStep = new AtomicLong(0);

    private static TemporalUnit DEFAULT_UNIT = ChronoUnit.MILLIS;
    private final TemporalUnit unit;

    public static void setDefaultUnit(TemporalUnit temporalUnit) {
        AverageTimeComponent.DEFAULT_UNIT = temporalUnit;
    }

    public AverageTimeComponent(TemporalUnit unit) {
        this.unit = unit;
    }

    public AverageTimeComponent() {
        this(DEFAULT_UNIT);
    }

    @Override
    public String render() {
        String result = elapsedTime.render();
        if (currentStep.get() != 0) {
            long averageStepTime = unit.between(elapsedTime.getStartTime(), Instant.now());
            result += " Average step time: " + averageStepTime + " " + unit.toString();
        }
        return result;
    }

    public void step() {
        currentStep.getAndIncrement();
    }
}
