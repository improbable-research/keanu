package io.improbable.keanu.network.grouping.continuouspointgroupers;

import static java.util.stream.Collectors.toList;

import io.improbable.keanu.network.grouping.ContinuousPoint;
import java.util.List;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

public class DBSCANContinuousPointGrouper implements ContinuousPointGrouper {

    private double eps;
    private int minPts;

    public DBSCANContinuousPointGrouper(double eps, int minPts) {
        this.eps = eps;
        this.minPts = minPts;
    }

    public List<List<ContinuousPoint>> groupContinuousPoints(List<ContinuousPoint> points) {

        DBSCANClusterer<ContinuousPoint> kmeansClusterer = new DBSCANClusterer<>(eps, minPts);

        List<Cluster<ContinuousPoint>> clusters = kmeansClusterer.cluster(points);

        return clusters.stream().map(Cluster::getPoints).collect(toList());
    }
}
