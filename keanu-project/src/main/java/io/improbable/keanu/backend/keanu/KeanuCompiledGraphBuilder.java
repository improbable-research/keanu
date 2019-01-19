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
    }

    private void startSource(StringBuilder sb) {

        sb.append("package io.improbable.keanu.backend.keanu;\n");
        sb.append("import io.improbable.keanu.backend.VariableReference;\n");
        sb.append("import java.util.Collection;\n");
        sb.append("import java.util.Collections;\n");
        sb.append("import java.util.HashMap;\n");
        sb.append("import java.util.Map;\n");
        sb.append("import io.improbable.keanu.tensor.dbl.DoubleTensor;\n");

        sb.append("public class " + className + " implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {\n");
        sb.append("public Map<String, ?> apply(Map<String, ?> inputs) {\n");
    }

    private void endSource(StringBuilder sb) {
        sb.append("Map<String, Object>  results = new HashMap<>();\n");
        sb.append("\n");

        for (VariableReference out : outputs) {
            String outputVariableName = toSourceVariableName(out);
            sb.append("results.put(\"" + out.toStringReference() + "\", " + outputVariableName + ");\n");
        }

        sb.append("return results;\n");

        sb.append("}\n}\n");
    }

    @Override
    public void createConstant(Vertex visiting) {

        Object value = visiting.getValue();
        String constantType = getType(value);
        String constantName = toSourceVariableName(visiting.getReference());

        declareInput(constantType, constantName, visiting.getReference().toStringReference());

        lookup.put(visiting.getReference(), constantName);
        constantValues.put(visiting.getReference(), value);

    }

    @Override
    public void createVariable(Vertex visiting) {

        Object value = visiting.getValue();
        String variableType = getType(value);
        String variableName = toSourceVariableName(visiting.getReference());

        declareInput(variableType, variableName, visiting.getReference().toStringReference());

        lookup.put(visiting.getReference(), variableName);
        variableValues.put(visiting.getReference(), value);
    }

    private void declareInput(String type, String name, String inputName) {
        sourceBuilder.append(type + " " + name + " = " + "(" + type + ")" + "inputs.get(\"" + inputName + "\");\n");
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

    public void registerOutput(Vertex output) {
        outputs.add(output.getReference());
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
        StringBuilder stringBuilder = new StringBuilder();

        startSource(stringBuilder);
        stringBuilder.append(sourceBuilder);
        endSource(stringBuilder);

        String source = stringBuilder.toString();
//        System.out.println(source);

        return compile(source);
    }

    private ComputableGraph compile(String source) {

        Function<Map<String, ?>, Map<String, ?>> computeFunction = Reflect.compile(
            "io.improbable.keanu.backend.keanu." + className,
            source
        ).create().get();

        return new ComputableGraph() {

            Map<String, VariableReference> outputsByString = outputs.stream()
                .collect(toMap(VariableReference::toStringReference, output -> output));

            @Override
            public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

                Map<String, Object> inputsByString = inputs.entrySet().stream()
                    .collect(toMap(e -> e.getKey().toStringReference(), Map.Entry::getValue));

                Map<String, Object> constantsByString = constantValues.entrySet().stream()
                    .collect(toMap(
                        e -> e.getKey().toStringReference(),
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
