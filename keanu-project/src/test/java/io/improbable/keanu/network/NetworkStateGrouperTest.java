package io.improbable.keanu.network;

import io.improbable.keanu.network.grouping.NetworkStateGrouper;
import io.improbable.keanu.network.grouping.continuouspointgroupers.DBSCANContinuousPointGrouper;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertTrue;

public class NetworkStateGrouperTest {

    private Random random = new Random(1);

    String v1Id = "v1";
    String v2Id = "v2";
    String v3Id = "v3";
    String v4Id = "v4";
    String v5Id = "v5";

    @Test
    public void finds5Groupings() {

        List<NetworkState> networkStates = new ArrayList<>();

        networkStates.addAll(createGroup(true, true, 1.0, 2.0, 3.0));
        networkStates.addAll(createGroup(true, true, 3.0, 2.0, 1.0));
        networkStates.addAll(createGroup(false, true, 4.0, 5.0, 6.0));
        networkStates.addAll(createGroup(true, false, 6.0, 5.0, 4.0));
        networkStates.addAll(createGroup(false, false, 100.0, 200.0, 50000.0));

        List<String> discreteIds = Arrays.asList(v1Id, v2Id);
        List<String> continuousIds = Arrays.asList(v3Id, v4Id, v5Id);

        NetworkStateGrouper grouper = new NetworkStateGrouper(new DBSCANContinuousPointGrouper(1.0, 3));
        List<List<NetworkState>> filteredStates = grouper.groupNetworkStates(networkStates, discreteIds, continuousIds);

        assertTrue(filteredStates.size() == 5);
    }

    private List<NetworkState> createGroup(boolean v1, boolean v2, double v3, double v4, double v5) {

        List<NetworkState> group = new ArrayList<>();

        for (int i = 0; i < 100; i++) {
            Map<String, ? super Object> values = new HashMap<>();
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
