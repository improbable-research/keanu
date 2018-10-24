package io.improbable.keanu.network.grouping;

import org.apache.commons.math3.ml.clustering.Clusterable;

import java.util.Arrays;

public class ContinuousPoint implements Clusterable {

    private final double[] point;

    public ContinuousPoint(double[] point) {
        this.point = Arrays.copyOf(point, point.length);
    }

    @Override
    public double[] getPoint() {
        return Arrays.copyOf(point, point.length);
    }
}
