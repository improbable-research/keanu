package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class TestGraph implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {

    private final DoubleTensor v_4;

    public TestGraph(Map<String, ?> constants) {

        v_4 = (DoubleTensor) constants.get("4");
    }

    public Map<String, ?> apply(Map<String, ?> inputs) {

        DoubleTensor v_0 = (DoubleTensor) inputs.get("0");
        DoubleTensor v_1 = (DoubleTensor) inputs.get("1");

        DoubleTensor v_2 = (DoubleTensor) v_0.times(v_1);

        DoubleTensor v_3 = (DoubleTensor) v_2.plus(v_4);

        Map<String, Object> results = new HashMap<>();
        results.put("3", v_3);

        return results;
    }

}
