package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class TestGraph implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {

    public Map<String, ?> apply(Map<String, ?> inputs) {

        DoubleTensor A = (DoubleTensor) inputs.get("A");
        DoubleTensor B = (DoubleTensor) inputs.get("B");

        DoubleTensor C = (DoubleTensor) A.times(B);

        Map<String, Object> results = new HashMap<>();
        results.put("C", C);

        return results;
    }

}

