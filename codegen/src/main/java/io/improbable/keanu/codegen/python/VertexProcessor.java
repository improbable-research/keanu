package io.improbable.keanu.codegen.python;

import com.google.common.base.CaseFormat;
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

            String[] pythonParameters = reflections.getConstructorParamNames(constructor).stream().map(
                parameter -> CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, parameter)).toArray(String[]::new);

            Class<?>[] parameterTypes = constructor.getParameterTypes();

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));

            PythonConstructor pythonConstructor = new PythonConstructor(
                javaClass,
                toPythonClass(javaClass),
                toTypedPythonParams(pythonParameters, parameterTypes),
                toCastedPythonParams(pythonParameters, parameterTypes),
                docString.getAsString()
            );

            pythonConstructors.add(pythonConstructor);
            exportedMethodsJoiner.add(pythonConstructor.pythonClass);
        }
        root.put("exportedMethods", exportedMethodsJoiner.toString());

        return root;
    }

    private static String toCast(Class<?> aClass) {
        if (DoubleVertex.class.isAssignableFrom(aClass)) {
            return "Double";
        } else if (IntegerVertex.class.isAssignableFrom(aClass)) {
            return "Integer";
        } else if (BoolVertex.class.isAssignableFrom(aClass)) {
            return "Bool";
        } else {
            throw new NotImplementedException(aClass.getCanonicalName());
        }
    }

    private static String toCastedPythonParams(String[] pythonParameters, Class<?>[] parameterTypes) {
        String[] pythonParams = new String[pythonParameters.length];

        for (int i = 0; i < pythonParameters.length; i++) {
            pythonParams[i] = toCastedPythonParam(pythonParameters[i], parameterTypes[i]);
        }

        return String.join(", ", pythonParams);
    }

    private static String toCastedPythonParam(String pythonParameter, Class<?> parameterType) {
        if (DoubleVertex.class.isAssignableFrom(parameterType)) {
            return "cast_double(" + pythonParameter + ")";
        } else if (IntegerVertex.class.isAssignableFrom(parameterType)) {
            return "cast_integer(" + pythonParameter + ")";
        } else if (BoolVertex.class.isAssignableFrom(parameterType)) {
            return "cast_bool(" + pythonParameter + ")";
        } else {
            return pythonParameter;
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
            return "vertex_param_types";
        } else if (DoubleTensor.class.isAssignableFrom(parameterType) ||
                   IntegerTensor.class.isAssignableFrom(parameterType) ||
                   BooleanTensor.class.isAssignableFrom(parameterType)) {
            return "tensor_arg_types";
        } else if (parameterType.isArray()) {
            return "shape_types";
        } else {
            throw new NotImplementedException(String.format("Mapping from Java type %s is not defined.", parameterType.getName()));
        }
    }

    private static List<Constructor> getSortedListOfAnnotatedVertexConstructors(Reflections reflections) {
        List<Constructor> constructors = new ArrayList<>(reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class));
        constructors.sort(Comparator.comparing(Constructor::getName));

        return constructors;
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
        private String pythonTypedParameters;
        @Getter
        private String pythonParameters;
        @Getter
        private String docString;

        PythonConstructor(String javaClass, String pythonClass, String pythonTypedParameters, String pythonParameters, String docString) {
            this.javaClass = javaClass;
            this.pythonClass = pythonClass;
            this.pythonTypedParameters = pythonTypedParameters;
            this.pythonParameters = pythonParameters;
            this.docString = docString;
        }
    }

}
