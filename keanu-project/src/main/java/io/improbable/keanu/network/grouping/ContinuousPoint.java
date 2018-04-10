package io.improbable.keanu.network.grouping;

import org.apache.commons.math3.ml.clustering.Clusterable;

public class ContinuousPoint implements Clusterable {

    private final double[] point;

    public ContinuousPoint(double[] point) {
        this.point = point;
    }

    @Override
    public double[] getPoint() {
        return point;
    }
}
