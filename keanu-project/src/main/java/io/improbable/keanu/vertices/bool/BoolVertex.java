package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.vertices.DiscreteVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.IfVertex;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public abstract class BoolVertex extends DiscreteVertex<Boolean> {

    @SafeVarargs
    public final BoolVertex or(Vertex<Boolean>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @SafeVarargs
    public final BoolVertex and(Vertex<Boolean>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        return new AndMultipleVertex(inputList(those));
    }

    private List<Vertex<Boolean>> inputList(Vertex<Boolean>[] those) {
        List<Vertex<Boolean>> inputs = new LinkedList<>();
        inputs.addAll(Arrays.asList(those));
        inputs.add(this);
        return inputs;
    }

    public static <T> Vertex<T> If(Vertex<Boolean> predicate, Vertex<T> thn, Vertex<T> els) {
        return new IfVertex<>(predicate, thn, els);
    }

}
