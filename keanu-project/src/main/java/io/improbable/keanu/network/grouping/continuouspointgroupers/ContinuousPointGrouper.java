package io.improbable.keanu.network.grouping.continuouspointgroupers;

import io.improbable.keanu.network.grouping.ContinuousPoint;
import java.util.List;

public interface ContinuousPointGrouper {
  List<List<ContinuousPoint>> groupContinuousPoints(List<ContinuousPoint> points);
}
