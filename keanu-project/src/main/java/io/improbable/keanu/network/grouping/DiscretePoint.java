package io.improbable.keanu.network.grouping;

import java.util.Arrays;

public class DiscretePoint {

    private final Object[] point;

    public DiscretePoint(Object[] point) {
        this.point = Arrays.copyOf(point, point.length);
    }

    public Object[] getPoint() {
        return Arrays.copyOf(point, point.length);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DiscretePoint that = (DiscretePoint) o;
        return Arrays.equals(point, that.point);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(point);
    }
}
