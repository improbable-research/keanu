package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraphBuilder;
import io.improbable.keanu.backend.StringVariableReference;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Scope;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class TensorflowComputableGraphBuilder implements ComputableGraphBuilder<TensorflowComputableGraph> {

    @Getter
    private final Map<VariableReference, Output<?>> lookup;

    private final Map<VariableReference, Object> variableValues;
    private final TensorflowOpFactory opFactory;
    private final Scope scope;

    public TensorflowComputableGraphBuilder() {
        lookup = new HashMap<>();
        variableValues = new HashMap<>();
        scope = new Scope(new Graph());
        opFactory = new TensorflowOpFactory(scope);
    }

    @Override
    public void createConstant(Vertex visiting) {

        Output<?> converted = KeanuToTensorflowOpMapper.createConstant(visiting, opFactory);
        lookup.put(visiting.getReference(), converted);
    }

    @Override
    public void createVariable(Vertex visiting) {

        Output<?> converted = KeanuToTensorflowOpMapper.createVariable(visiting, opFactory);
        lookup.put(visiting.getReference(), converted);
        variableValues.put(visiting.getReference(), visiting.getValue());
    }

    @Override
    public Collection<VariableReference> getLatentVariables() {
        return variableValues.keySet();
    }

    @Override
    public void create(Vertex visiting) {

        KeanuToTensorflowOpMapper.OpMapper opMapper = KeanuToTensorflowOpMapper.getOpMapperFor(visiting.getClass());

        if (opMapper == null) {
            throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported for Tensorflow conversion");
        }

        Output<?> converted = opMapper.apply(visiting, lookup, opFactory);
        lookup.put(visiting.getReference(), converted);
    }

    @Override
    public void connect(Map<Vertex<?>, Vertex<?>> connections) {
        connections.forEach((to, from) ->
            lookup.put(from.getReference(), lookup.get(to.getReference()))
        );
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {
        Output<?> sum = opFactory.add((Output<Double>) lookup.get(left), (Output<Double>) lookup.get(right));
        VariableReference sumReference = new StringVariableReference(sum.op().name());

        lookup.put(sumReference, sum);

        return sumReference;
    }

    @Override
    public TensorflowComputableGraph build() {
        return new TensorflowComputableGraph(new Session(scope.graph()), scope, variableValues);
    }
}
