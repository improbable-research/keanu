package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraphBuilder;
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

public class TensorflowGraphBuilder implements ComputableGraphBuilder<TensorflowComputableGraph> {

    @Getter
    private final Map<VariableReference, Output<?>> lookup;

    private final Map<VariableReference, Object> variableValues;
    private final TensorflowOpHelper opHelper;
    private final Scope scope;

    public TensorflowGraphBuilder() {
        lookup = new HashMap<>();
        variableValues = new HashMap<>();
        scope = new Scope(new Graph());
        opHelper = new TensorflowOpHelper(scope);
    }

    @Override
    public void createConstant(Vertex visiting) {

        Output<?> converted = TensorflowGraphConverter.createConstant(visiting, opHelper);
        lookup.put(visiting.getReference(), converted);
    }

    @Override
    public void createVariable(Vertex visiting) {

        Output<?> converted = TensorflowGraphConverter.createVariable(visiting, opHelper);
        lookup.put(visiting.getReference(), converted);
        variableValues.put(visiting.getReference(), visiting.getValue());
    }

    @Override
    public Collection<VariableReference> getLatentVariables() {
        return variableValues.keySet();
    }

    @Override
    public void create(Vertex visiting) {

        TensorflowGraphConverter.OpMapper opMapper = TensorflowGraphConverter.opMappers.get(visiting.getClass());

        if (opMapper == null) {
            throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported for Tensorflow conversion");
        }

        Output<?> converted = opMapper.apply(visiting, lookup, opHelper);
        lookup.put(visiting.getReference(), converted);
    }

    @Override
    public void alias(VariableReference from, VariableReference to) {
        lookup.put(from, lookup.get(to));
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {
        Output<?> sum = opHelper.add((Output<Double>) lookup.get(left), (Output<Double>) lookup.get(right));
        VariableReference sumReference = new StringVariableReference(sum.op().name());

        lookup.put(sumReference, sum);

        return sumReference;
    }

    @Override
    public TensorflowComputableGraph build() {
        return new TensorflowComputableGraph(new Session(scope.graph()), scope, variableValues);
    }
}
