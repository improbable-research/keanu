package io.improbable.keanu.codegen.python;

import com.google.common.base.CaseFormat;
import com.google.gson.internal.Primitives;
import freemarker.template.Template;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import lombok.Getter;
import org.apache.commons.lang3.NotImplementedException;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.scanners.MethodParameterNamesScanner;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import java.util.stream.Collectors;

class VertexProcessor {

    final private static String TEMPLATE_FILE = "generated.py.ftl";
    final private static String GENERATED_FILE = "generated.py";
    final private static String TEMPLATE_INIT_FILE = "__init__.py.ftl";
    final private static String GENERATED_INIT_FILE = "__init__.py";

    static void process(String generatedDir) throws IOException {
        Map<String, Object> dataModel = buildDataModel();
        Template generatedVerticesFileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_FILE);
        Writer fileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_FILE);
        Template generatedInitFileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_INIT_FILE);
        Writer initFileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_INIT_FILE);

        TemplateProcessor.processDataModel(dataModel, generatedVerticesFileTemplate, fileWriter);
        TemplateProcessor.processDataModel(dataModel, generatedInitFileTemplate, initFileWriter);
    }

    private static Map<String, Object> buildDataModel() throws IOException {
        Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner(), new MethodParameterNamesScanner()));

        List<Constructor> constructors = getSortedListOfAnnotatedVertexConstructors(reflections);

        Map<String, Object> root = new HashMap<>();
        List<Import> imports = new ArrayList<>();
        List<PythonConstructor> pythonConstructors = new ArrayList<>();

        root.put("imports", imports);
        root.put("constructors", pythonConstructors);
        Map<String, DocString> nameToDocStringMap = KeanuProjectDoclet.getDocStringsFromFile();
        StringJoiner exportedMethodsJoiner = new StringJoiner("\", \"", "\"", "\"");

        for (Constructor constructor : constructors) {
            String javaClass = constructor.getDeclaringClass().getSimpleName();
            String qualifiedName = constructor.getName();
            DocString docString = nameToDocStringMap.get(qualifiedName);

            String[] parametersWithPythonFormatting = reflections.getConstructorParamNames(constructor).stream().map(
                parameter -> CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, parameter)).toArray(String[]::new);

            Class<?>[] parameterTypes = constructor.getParameterTypes();

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));

            PythonConstructor pythonConstructor = new PythonConstructor(
                javaClass,
                toPythonClass(javaClass),
                toPythonVertexClass(constructor.getDeclaringClass()),
                toTypedPythonParams(parametersWithPythonFormatting, parameterTypes),
                toCastedPythonParams(parametersWithPythonFormatting, parameterTypes),
                docString.getAsString()
            );

            pythonConstructors.add(pythonConstructor);
            exportedMethodsJoiner.add(pythonConstructor.pythonClass);
        }
        root.put("exportedMethods", exportedMethodsJoiner.toString());

        return root;
    }

    private static String toPythonVertexClass(Class<?> javaClass) {
        if (DoubleVertex.class.isAssignableFrom(javaClass)) {
            return "Double";
        } else if (IntegerVertex.class.isAssignableFrom(javaClass)) {
            return "Integer";
        } else if (BoolVertex.class.isAssignableFrom(javaClass)) {
            return "Bool";
        } else {
            return "Vertex";
        }
    }

    private static String toCastedPythonParams(String[] pythonParameters, Class<?>[] parameterTypes) {
        String[] pythonParams = new String[pythonParameters.length];

        for (int i = 0; i < pythonParameters.length; i++) {
            pythonParams[i] = toCastedPythonParam(pythonParameters[i], parameterTypes[i]);
        }

        return String.join(", ", pythonParams);
    }

    private static String toCastedPythonParam(String pythonParameter, Class<?> parameterClass) {
        Class parameterType = Primitives.wrap(parameterClass);

        if (DoubleVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_double_vertex(" + pythonParameter + ")";
        } else if (IntegerVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer_vertex(" + pythonParameter + ")";
        } else if (BoolVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_boolean_vertex(" + pythonParameter + ")";
        } else if (Vertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_vertex(" + pythonParameter + ")";
        } else if (DoubleTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_double_tensor(" + pythonParameter + ")";
        } else if (IntegerTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer_tensor(" + pythonParameter + ")";
        } else if (BooleanTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_boolean_tensor(" + pythonParameter + ")";
        } else if (Double.class.isAssignableFrom(parameterType)) {
            return "cast_to_double(" + pythonParameter + ")";
        } else if (Integer.class.isAssignableFrom(parameterType) || Long.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer(" + pythonParameter + ")";
        } else if (String.class.isAssignableFrom(parameterType)) {
            return "cast_to_string(" + pythonParameter + ")";
        } else if (Long[].class.isAssignableFrom(parameterType) || Integer[].class.isAssignableFrom(parameterType) ||
            long[].class.isAssignableFrom(parameterType) || int[].class.isAssignableFrom(parameterType)) {
            //TODO - This should only be for longs, and should be called array
            return "cast_to_long_list(" + pythonParameter + ")";
        } else if (Vertex[].class.isAssignableFrom(parameterType)) {
            return "cast_to_vertex_list(" + pythonParameter + ")";
        } else {
            throw new IllegalArgumentException("Failed to Encode " + pythonParameter + " of type: " + parameterType);
        }
    }

    private static String toTypedPythonParams(String[] pythonParameters, Class<?>[] parameterTypes) {
        String[] pythonParams = new String[pythonParameters.length];

        for (int i = 0; i < pythonParameters.length; i++) {
            pythonParams[i] = pythonParameters[i] + ": " + toTypedPythonParam(parameterTypes[i]);
        }

        return String.join(", ", pythonParams);
    }

    private static String toTypedPythonParam(Class<?> parameterType) {
        if (Vertex.class.isAssignableFrom(parameterType)) {
            return "vertex_constructor_param_types";
        } else if (DoubleTensor.class.isAssignableFrom(parameterType) ||
                   IntegerTensor.class.isAssignableFrom(parameterType) ||
                   BooleanTensor.class.isAssignableFrom(parameterType)) {
            return "tensor_arg_types";
        } else if (Double.class.isAssignableFrom(parameterType)) {
            return "float";
        } else if (Integer.class.isAssignableFrom(parameterType) || Long.class.isAssignableFrom(parameterType)) {
            return "int";
        } else if (String.class.isAssignableFrom(parameterType)) {
            return "str";
        } else if (Long[].class.isAssignableFrom(parameterType) || Integer[].class.isAssignableFrom(parameterType) ||
            long[].class.isAssignableFrom(parameterType) || int[].class.isAssignableFrom(parameterType)) {
            return "Iterable[int]";
        } else if (Vertex[].class.isAssignableFrom(parameterType)) {
            return "Iterable[vertex_constructor_param_types]";
        } else {
            throw new NotImplementedException(String.format("Mapping from Java type %s is not defined.", parameterType.getName()));
        }
    }

    private static List<Constructor> getSortedListOfAnnotatedVertexConstructors(Reflections reflections) {
        return reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class).stream()
            .sorted(Comparator.comparing(Constructor::getName))
            .collect(Collectors.toList());
    }

    private static String toPythonClass(String javaClass) {
        return javaClass.replaceAll("Vertex$", "");
    }

    public static class Import {
        @Getter
        private String packageName;

        Import(String packageName) {
            this.packageName = packageName;
        }
    }

    public static class PythonConstructor {
        @Getter
        private String javaClass;
        @Getter
        private String pythonClass;
        @Getter
        private String pythonVertexClass;
        @Getter
        private String pythonTypedParameters;
        @Getter
        private String pythonParameters;
        @Getter
        private String docString;

        PythonConstructor(String javaClass, String pythonClass, String pythonVertexClass, String pythonTypedParameters, String pythonParameters, String docString) {
            this.javaClass = javaClass;
            this.pythonClass = pythonClass;
            this.pythonVertexClass = pythonVertexClass;
            this.pythonTypedParameters = pythonTypedParameters;
            this.pythonParameters = pythonParameters;
            this.docString = docString;
        }
    }

}
