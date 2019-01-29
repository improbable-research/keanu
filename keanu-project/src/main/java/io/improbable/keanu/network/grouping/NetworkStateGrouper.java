package io.improbable.keanu.network.grouping;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.network.grouping.continuouspointgroupers.ContinuousPointGrouper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

/**
 * This class has the ability to group NetworkStates that are similar. Which NetworkStates
 * are similar are determined by a ContinuousPointGrouper (K-means-like algorithm).
 * ALL unique discrete states will be considered a group and all associated
 * continuous states (i.e. double) will be grouped with the ContinuousPointGrouper,
 * which is defined by the user.
 */
public class NetworkStateGrouper {

    private final ContinuousPointGrouper continuousPointGrouper;

    public NetworkStateGrouper(ContinuousPointGrouper continuousPointGrouper) {
        this.continuousPointGrouper = continuousPointGrouper;
    }

    /**
     * @param networkStates       the NetworkStates to be grouped
     * @param discreteVertexIds   ids of vertices that do NOT contain Double values
     * @param continuousVertexIds ids of vertices that contain Double values
     * @return a list of lists of NetworkStates where each sub list is a grouped subset
     * of the supplied networkStates
     */
    public List<List<NetworkState>> groupNetworkStates(List<NetworkState> networkStates,
                                                       List<VariableReference> discreteVertexIds,
                                                       List<VariableReference> continuousVertexIds) {


        Map<DiscretePoint, List<NetworkState>> statesGroupedByDiscretePoint = networkStates.stream()
            .collect(groupingBy(state -> toDiscretePoint(state, discreteVertexIds)));

        Map<DiscretePoint, List<List<ContinuousPoint>>> continuousPointsGroupedByDiscretePoint = statesGroupedByDiscretePoint.entrySet().stream()
            .collect(
                toMap(
                    Map.Entry::getKey,
                    e -> continuousPointGrouper.groupContinuousPoints(
                        toContinuousPoints(e.getValue(), continuousVertexIds)
                    )
                )
            );

        return continuousPointsGroupedByDiscretePoint.entrySet().stream()
            .flatMap(e -> toListOfNetworkStates(e.getKey(), e.getValue(), discreteVertexIds, continuousVertexIds))
            .collect(toList());
    }

    private Stream<List<NetworkState>> toListOfNetworkStates(DiscretePoint discretePoint,
                                                             List<List<ContinuousPoint>> continuousPoints,
                                                             List<VariableReference> discreteVertexIds,
                                                             List<VariableReference> continuousVertexIds) {

        Map<VariableReference, ?> discreteValues = fromDiscretePoint(discretePoint, discreteVertexIds);

        return continuousPoints.stream()
            .map(groupedPoints -> toNetworkState(discreteValues, groupedPoints, continuousVertexIds));
    }

    private List<NetworkState> toNetworkState(Map<VariableReference, ?> discreteValues, List<ContinuousPoint> continuousPoints, List<VariableReference> continuousVertexIds) {
        return continuousPoints.stream().map(point -> {
            Map<VariableReference, ? super Object> networkState = new HashMap<>();
            Map<VariableReference, Double> continuousValues = fromContinuousPoint(point, continuousVertexIds);
            networkState.putAll(continuousValues);
            networkState.putAll(discreteValues);
            return new SimpleNetworkState(networkState);
        }).collect(toList());
    }

    private DiscretePoint toDiscretePoint(NetworkState networkState, List<VariableReference> discreteVertexIds) {
        Object[] point = new Object[discreteVertexIds.size()];
        for (int vertex = 0; vertex < discreteVertexIds.size(); vertex++) {
            point[vertex] = networkState.get(discreteVertexIds.get(vertex));
        }
        return new DiscretePoint(point);
    }

    private Map<VariableReference, ?> fromDiscretePoint(DiscretePoint discretePoint, List<VariableReference> discreteVertexIds) {
        Map<VariableReference, ? super Object> discreteStates = new HashMap<>();
        Object[] discreteValues = discretePoint.getPoint();
        for (int i = 0; i < discreteVertexIds.size(); i++) {
            discreteStates.put(discreteVertexIds.get(i), discreteValues[i]);
        }

        return discreteStates;
    }

    private List<ContinuousPoint> toContinuousPoints(List<NetworkState> networkStates, List<VariableReference> continuousVertexIds) {
        return networkStates.stream()
            .map(p -> toContinuousPoint(p, continuousVertexIds))
            .collect(toList());
    }

    private ContinuousPoint toContinuousPoint(NetworkState networkState, List<VariableReference> continuousVertexIds) {
        double[] point = new double[continuousVertexIds.size()];

        int i = 0;
        for (VariableReference vertexId : continuousVertexIds) {
            point[i] = networkState.get(vertexId);
            i++;
        }

        return new ContinuousPoint(point);
    }

    private Map<VariableReference, Double> fromContinuousPoint(ContinuousPoint point, List<VariableReference> continuousVertexIds) {
        Map<VariableReference, Double> continuousStates = new HashMap<>();
        double[] continuousValues = point.getPoint();
        for (int i = 0; i < continuousVertexIds.size(); i++) {
            continuousStates.put(continuousVertexIds.get(i), continuousValues[i]);
        }

        return continuousStates;
    }
}
