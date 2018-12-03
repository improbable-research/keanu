package io.improbable.keanu.network;

import io.improbable.keanu.vertices.VertexId;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SimpleNetworkStateTest {
    Map<VertexId, Double> sampleMap = new HashMap<>();

    @Before
    public void setup() {
        VertexId v1 = new VertexId();
        sampleMap.put(v1, 22.7);
    }

    @Test
    public void canGetLogOfMasterPWhenPresent() {
        final double expectedLogOfMasterP = 23.5;
        NetworkState networkState = new SimpleNetworkState(sampleMap, expectedLogOfMasterP);
        assertEquals(expectedLogOfMasterP, networkState.getLogOfMasterP(), 0.01);
    }

    @Test(expected = IllegalArgumentException.class)
    public void getLogOfMasterPThrowsExceptionWhenNotPresent() {
        NetworkState networkState = new SimpleNetworkState(sampleMap);
        networkState.getLogOfMasterP();
    }
}
