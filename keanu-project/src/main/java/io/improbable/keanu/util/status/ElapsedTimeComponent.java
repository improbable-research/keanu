package io.improbable.keanu.util.status;

import lombok.Getter;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;

public class ElapsedTimeComponent implements StatusBarComponent {
    @Getter
    private final Instant startTime = Instant.now();

    private static TemporalUnit DEFAULT_UNIT = ChronoUnit.SECONDS;
    private final TemporalUnit unit;

    public static void setDefaultUnit(TemporalUnit temporalUnit) {
        ElapsedTimeComponent.DEFAULT_UNIT = temporalUnit;
    }

    public ElapsedTimeComponent(TemporalUnit unit) {
        this.unit = unit;
    }

    public ElapsedTimeComponent() {
        this(DEFAULT_UNIT);
    }

    @Override
    public String render() {
        long timeDifference = unit.between(startTime, Instant.now());
        return "Elapsed time: " + timeDifference + " " + unit.toString();
    }
}
