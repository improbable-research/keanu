package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.ComputableGraphBuilder;
import io.improbable.keanu.backend.StringVariableReference;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;
import org.joor.Reflect;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

public class KeanuCompiledGraphBuilder implements ComputableGraphBuilder<ComputableGraph> {

    private static final String PACKAGE = "io.improbable.keanu.backend.keanu";
    private static final String CLASS_NAME_PREFIX = "CompiledKeanuGraph";

    private StringBuilder computeSourceBuilder;
    private StringBuilder instanceVariableBuilder;
    private StringBuilder constructorBuilder;
    private Map<VariableReference, KeanuCompiledVariable> lookup;
    private Map<VariableReference, Object> variableValues;
    private Map<VariableReference, Object> constantValues;
    private List<VariableReference> outputs;

    private int internalOpCount = 0;

    private final String className = CLASS_NAME_PREFIX + this.hashCode();

    public KeanuCompiledGraphBuilder() {
        computeSourceBuilder = new StringBuilder();
        instanceVariableBuilder = new StringBuilder();
        constructorBuilder = new StringBuilder();
        lookup = new HashMap<>();
        variableValues = new HashMap<>();
        constantValues = new HashMap<>();
        outputs = new ArrayList<>();
    }

    private void startSource(StringBuilder sb) {

        sb.append("package " + PACKAGE + ";\n");
        sb.append(importString(Collection.class));
        sb.append(importString(Collections.class));
        sb.append(importString(HashMap.class));
        sb.append(importString(Map.class));
        sb.append(importString(VariableReference.class));
        sb.append(importString(DoubleTensor.class));
        sb.append(importString(IntegerTensor.class));
        sb.append(importString(BooleanTensor.class));

        append(sb, "public final class ", className, " implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {\n");
    }

    private String importString(Class<?> clazz) {
        return "import " + clazz.getCanonicalName() + ";\n";
    }

    private void endSource(StringBuilder sb) {
        sb.append("Map<String, Object>  results = new HashMap<>();\n");
        sb.append("\n");

        for (VariableReference out : outputs) {
            String name = lookup.get(out).getName();
            append(sb, "results.put(\"", out.toStringReference(), "\", ", name, ");\n");
        }

        sb.append("return results;\n");

        sb.append("}\n}\n");
    }

    @Override
    public void createConstant(Vertex visiting) {

        String type = getAssigmentType(visiting);
        String lookupName = visiting.getReference().toStringReference();
        String name = toSourceVariableName(visiting.getReference());

        append(instanceVariableBuilder, "private final ", type, " ", name, ";\n");
        append(constructorBuilder, name, " = ", "(", type, ")", "constants.get(\"", lookupName, "\");\n");

        lookup.put(visiting.getReference(), new KeanuCompiledVariable(name, false));
        constantValues.put(visiting.getReference(), visiting.getValue());

    }

    @Override
    public void createVariable(Vertex visiting) {

        String variableType = getAssigmentType(visiting);
        String variableName = toSourceVariableName(visiting.getReference());

        declareInput(variableType, variableName, visiting.getReference().toStringReference());

        lookup.put(visiting.getReference(), new KeanuCompiledVariable(variableName, false));
        variableValues.put(visiting.getReference(), visiting.getValue());
    }

    private void declareInput(String type, String name, String inputName) {
        append(computeSourceBuilder, "final ", type, " ", name, " = (", type, ") inputs.get(\"", inputName, "\");\n");
    }

    @Override
    public void create(Vertex visiting) {

        if (isConstant(visiting)) {
            createConstant(visiting);
            return;
        }

        Vertex unwrappedVisiting = visiting instanceof VertexWrapper ? ((VertexWrapper) visiting).getWrappedVertex() : visiting;
        Class<?> clazz = unwrappedVisiting.getClass();
        KeanuVertexToTensorOpMapper.OpMapper opMapperFor = KeanuVertexToTensorOpMapper.getOpMapperFor(clazz);

        String variableType = getAssigmentType(visiting);
        String name = toSourceVariableName(visiting.getReference());

        append(computeSourceBuilder, "final ", variableType, " ", name, " = (", variableType, ") ", opMapperFor.apply(unwrappedVisiting, lookup), ";\n");

        lookup.put(visiting.getReference(), new KeanuCompiledVariable(name, true));
    }

    private boolean isConstant(Vertex v) {
        return v instanceof ConstantDoubleVertex || v instanceof ConstantIntegerVertex || v instanceof ConstantBooleanVertex;
    }

    private String getAssigmentType(Vertex v) {
        return v.ofType().getCanonicalName();
    }

    private String toSourceVariableName(VariableReference variableReference) {
        return "v_" + variableReference.toStringReference();
    }

    @Override
    public void registerOutput(VariableReference output) {
        outputs.add(output);
        lookup.get(output).setMutable(false);
    }

    @Override
    public Collection<VariableReference> getLatentVariables() {
        return variableValues.keySet();
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {

        String variableType = DoubleTensor.class.getCanonicalName();

        String leftName = lookup.get(left).getName();
        String rightName = lookup.get(right).getName();

        String name = "vv_" + internalOpCount;
        internalOpCount++;

        append(computeSourceBuilder, "final ", variableType, " ", name, " = ", leftName, ".plus(", rightName + ");\n");

        StringVariableReference reference = new StringVariableReference(name);
        lookup.put(reference, new KeanuCompiledVariable(name, true));

        return reference;
    }

    @Override
    public void connect(Map<? extends Vertex<?, ?>, ? extends Vertex<?, ?>> connections) {
        connections.forEach((to, from) ->
            lookup.put(from.getReference(), lookup.get(to.getReference()))
        );
    }

    /**
     * @return The java source to be used for compilation. This will be the entire .java file that represents a class
     * that can be used for doing a calculation described by the graph that was passed to this builder.
     */
    public String getSource() {
        StringBuilder stringBuilder = new StringBuilder();

        startSource(stringBuilder);

        stringBuilder.append(instanceVariableBuilder);

        append(stringBuilder, "public ", className, "(final Map<String, ?> constants) {\n");

        stringBuilder.append(constructorBuilder);
        stringBuilder.append("}\n");

        stringBuilder.append("public Map<String, ?> apply(Map<String, ?> inputs) {\n");
        stringBuilder.append(computeSourceBuilder);

        endSource(stringBuilder);

        return stringBuilder.toString();
    }

    private void append(StringBuilder sb, String... line) {
        for (String token : line) {
            sb.append(token);
        }
    }

    @Override
    public WrappedCompiledGraph build() {

        String source = getSource();

//        System.out.println(source);

        return compile(source);
    }

    private WrappedCompiledGraph compile(String source) {

        Map<String, ?> constantsByString = constantValues.entrySet().stream()
            .collect(toMap(e -> e.getKey().toStringReference(), Map.Entry::getValue));

        Function<Map<String, ?>, Map<String, ?>> computeFunction = Reflect.compile(
            PACKAGE + "." + className,
            source
        ).create(constantsByString).get();

        return new WrappedCompiledGraph(computeFunction, outputs);
    }

}
