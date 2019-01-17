package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.ComputableGraphBuilder;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.joor.Reflect;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

public class KeanuCompiledGraphBuilder implements ComputableGraphBuilder<ComputableGraph> {

    private StringBuilder sourceBuilder;
    private Map<VariableReference, String> lookup;
    private Map<VariableReference, Object> variableValues;
    private Map<VariableReference, Object> constantValues;
    private List<VariableReference> outputs;

    private final String className = "CompiledKeanuGraph" + this.hashCode();

    public KeanuCompiledGraphBuilder() {
        sourceBuilder = new StringBuilder();
        lookup = new HashMap<>();
        variableValues = new HashMap<>();
        constantValues = new HashMap<>();
        outputs = new ArrayList<>();
        startSource();
    }

    private void startSource() {

        sourceBuilder.append("package io.improbable.keanu.backend.keanu;\n");
        sourceBuilder.append("import io.improbable.keanu.backend.VariableReference;\n");
        sourceBuilder.append("import java.util.Collection;\n");
        sourceBuilder.append("import java.util.Collections;\n");
        sourceBuilder.append("import java.util.HashMap;\n");
        sourceBuilder.append("import java.util.Map;\n");
        sourceBuilder.append("import io.improbable.keanu.tensor.dbl.DoubleTensor;\n");

        sourceBuilder.append("public class " + className + " implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {\n");
        sourceBuilder.append("public Map<String, ?> apply(Map<String, ?> inputs) {\n");
    }

    private void endSource() {
        sourceBuilder.append("Map<String, Object>  results = new HashMap<>();\n");
        sourceBuilder.append("\n");

        for (VariableReference out : outputs) {
            String outputVariableName = toSourceVariableName(out);
            sourceBuilder.append("results.put(\"" + outputVariableName + "\", " + outputVariableName + ");\n");
        }

        sourceBuilder.append("return results;\n");

        sourceBuilder.append("}\n}\n");
    }

    @Override
    public void createConstant(Vertex visiting) {

        Object value = visiting.getValue();
        String constantType = getType(value);
        String constantName = toSourceVariableName(visiting.getReference());

        declareInput(constantType, constantName);

        lookup.put(visiting.getReference(), constantName);
        constantValues.put(visiting.getReference(), value);

    }

    @Override
    public void createVariable(Vertex visiting) {

        Object value = visiting.getValue();
        String variableType = getType(value);
        String variableName = toSourceVariableName(visiting.getReference());

        declareInput(variableType, variableName);

        lookup.put(visiting.getReference(), variableName);
        variableValues.put(visiting.getReference(), value);
    }

    private void declareInput(String type, String name) {
        sourceBuilder.append(type + " " + name + " = " + "(" + type + ")" + "inputs.get(\"" + name + "\");\n");
    }

    @Override
    public void create(Vertex visiting) {

        if (isConstant(visiting)) {
            createConstant(visiting);
            return;
        }

        KeanuVertexToTensorOpMapper.OpMapper opMapperFor = KeanuVertexToTensorOpMapper.getOpMapperFor(visiting.getClass());

        String variableType = getType(visiting);
        String name = toSourceVariableName(visiting.getReference());
        sourceBuilder.append(variableType + " " + name + " = " + opMapperFor.apply(visiting, lookup) + ";\n");

        lookup.put(visiting.getReference(), name);
    }

    private boolean isConstant(Vertex v) {
        return v instanceof ConstantDoubleVertex || v instanceof ConstantIntegerVertex || v instanceof ConstantBooleanVertex;
    }

    private String getType(Object v) {
        return "DoubleTensor";
    }

    private String toSourceVariableName(VariableReference variableReference) {
        return "v_" + variableReference.toStringReference();
    }

    public void registerOutput(VariableReference variableReference) {
        outputs.add(variableReference);
    }

    @Override
    public Collection<VariableReference> getLatentVariables() {
        return variableValues.keySet();
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {

        return null;
    }

    @Override
    public void connect(Map<? extends Vertex<?>, ? extends Vertex<?>> connections) {
        connections.forEach((to, from) ->
            lookup.put(from.getReference(), lookup.get(to.getReference()))
        );
    }

    @Override
    public ComputableGraph build() {
        endSource();
        return compile();
    }

    private ComputableGraph compile() {

        String source = sourceBuilder.toString();

//        System.out.println(source);

        Function<Map<String, ?>, Map<String, ?>> computeFunction = Reflect.compile(
            "io.improbable.keanu.backend.keanu." + className,
            source
        ).create().get();

        return new ComputableGraph() {

            Map<String, VariableReference> outputsByString = KeanuCompiledGraphBuilder.this.outputs.stream()
                .collect(toMap(output -> toSourceVariableName(output), output -> output));

            @Override
            public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

                Map<String, Object> inputsByString = inputs.entrySet().stream()
                    .collect(toMap(e -> toSourceVariableName(e.getKey()), Map.Entry::getValue));

                Map<String, Object> constantsByString = constantValues.entrySet().stream()
                    .collect(toMap(
                        e -> toSourceVariableName(e.getKey()),
                        Map.Entry::getValue)
                    );

                inputsByString.putAll(constantsByString);

                Map<String, ?> results = computeFunction.apply(inputsByString);

                Map<VariableReference, ?> computed = results.entrySet().stream()
                    .collect(toMap(
                        e -> outputsByString.get(e.getKey()),
                        Map.Entry::getValue)
                    );

                return computed;
            }

            @Override
            public <T> T getInput(VariableReference input) {
                return null;
            }
        };
    }
}
