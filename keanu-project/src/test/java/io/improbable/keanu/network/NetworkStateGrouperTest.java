package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.grouping.NetworkStateGrouper;
import io.improbable.keanu.network.grouping.continuouspointgroupers.DBSCANContinuousPointGrouper;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class NetworkStateGrouperTest {

    private KeanuRandom random = new KeanuRandom(1);

    VertexId v1Id = new VertexId();
    VertexId v2Id = new VertexId();
    VertexId v3Id = new VertexId();
    VertexId v4Id = new VertexId();
    VertexId v5Id = new VertexId();

    @Test
    public void finds5Groupings() {

        List<NetworkState> networkStates = new ArrayList<>();

        networkStates.addAll(createGroup(true, true, 1.0, 2.0, 3.0));
        networkStates.addAll(createGroup(true, true, 3.0, 2.0, 1.0));
        networkStates.addAll(createGroup(false, true, 4.0, 5.0, 6.0));
        networkStates.addAll(createGroup(true, false, 6.0, 5.0, 4.0));
        networkStates.addAll(createGroup(false, false, 100.0, 200.0, 50000.0));

        List<VariableReference> discreteIds = Arrays.asList(v1Id, v2Id);
        List<VariableReference> continuousIds = Arrays.asList(v3Id, v4Id, v5Id);

        NetworkStateGrouper grouper = new NetworkStateGrouper(new DBSCANContinuousPointGrouper(1.0, 3));
        List<List<NetworkState>> filteredStates = grouper.groupNetworkStates(networkStates, discreteIds, continuousIds);

        assertTrue(filteredStates.size() == 5);
    }

    private List<NetworkState> createGroup(boolean v1, boolean v2, double v3, double v4, double v5) {

        List<NetworkState> group = new ArrayList<>();

        for (int i = 0; i < 100; i++) {
            Map<VariableReference, ? super Object> values = new HashMap<>();
            values.put(v1Id, v1);
            values.put(v2Id, v2);
            values.put(v3Id, v3 + (random.nextDouble() - 0.5));
            values.put(v4Id, v4 + (random.nextDouble() - 0.5));
            values.put(v5Id, v5 + (random.nextDouble() - 0.5));
            group.add(new SimpleNetworkState(values));
        }

        return group;
    }
}
