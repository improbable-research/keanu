package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.GraphBuilder;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Scope;

import java.util.HashMap;
import java.util.Map;

public class TensorflowGraphBuilder implements GraphBuilder<TensorflowComputableGraph> {

    @Getter
    private final Map<Vertex<?>, Output<?>> lookup;

    private final Map<VariableReference, Object> latentVariables;
    private final TensorflowOpHelper opHelper;
    private final Scope scope;

    public TensorflowGraphBuilder() {
        lookup = new HashMap<>();
        latentVariables = new HashMap<>();
        scope = new Scope(new Graph());
        opHelper = new TensorflowOpHelper(scope);
    }

    @Override
    public void createConstant(Vertex visiting) {

        Output<?> converted = TensorflowGraphConverter.createConstant(visiting, opHelper);
        lookup.put(visiting, converted);
    }

    @Override
    public void createVariable(Vertex visiting) {

        Output<?> converted = TensorflowGraphConverter.createVariable(visiting, opHelper);
        lookup.put(visiting, converted);
        latentVariables.put(visiting.getReference(), visiting.getValue());
    }

    @Override
    public void convert(Vertex visiting) {

        TensorflowGraphConverter.OpMapper opMapper = TensorflowGraphConverter.opMappers.get(visiting.getClass());

        if (opMapper == null) {
            throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported for Tensorflow conversion");
        }

        Output<?> converted = opMapper.apply(visiting, lookup, opHelper);
        lookup.put(visiting, converted);
    }

    @Override
    public TensorflowComputableGraph build() {
        return new TensorflowComputableGraph(new Session(scope.graph()), scope, latentVariables);
    }
}
