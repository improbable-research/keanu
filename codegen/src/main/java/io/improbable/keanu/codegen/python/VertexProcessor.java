package io.improbable.keanu.codegen.python;

import com.google.common.base.CaseFormat;
import freemarker.template.Template;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;
import org.apache.commons.lang3.NotImplementedException;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.scanners.MethodParameterNamesScanner;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.io.Writer;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class VertexProcessor {

    final private static String TEMPLATE_FILE = "generated.py.ftl";
    final private static String GENERATED_FILE = "generated.py";

    static void process(String generatedDir) {
        Map<String, Object> dataModel = buildDataModel();
        Template fileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_FILE);
        Writer fileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_FILE);

        TemplateProcessor.processDataModel(dataModel, fileTemplate, fileWriter);
    }

    private static Map<String, Object> buildDataModel() {
        Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner(), new MethodParameterNamesScanner()));

        List<Constructor> constructors = getSortedListOfAnnotatedVertexConstructors(reflections);

        Map<String, Object> root = new HashMap<>();
        List<Import> imports = new ArrayList<>();
        List<PythonConstructor> pythonConstructors = new ArrayList<>();

        root.put("imports", imports);
        root.put("constructors", pythonConstructors);

        for (Constructor constructor : constructors) {
            String javaClass = constructor.getDeclaringClass().getSimpleName();
            String[] pythonParameters = reflections.getConstructorParamNames(constructor).stream().map(
                parameter -> CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, parameter)).toArray(String[]::new);

            Class<?>[] parameterTypes = constructor.getParameterTypes();

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));
            pythonConstructors.add(
                new PythonConstructor(
                    javaClass,
                    toPythonClass(javaClass),
                    toPythonParams(pythonParameters, parameterTypes),
                    String.join(", ", pythonParameters)));
        }

        return root;
    }

    private static String toPythonParams(String[] pythonParameters, Class<?>[] parameterTypes) {
        String[] pythonParams = new String[pythonParameters.length];

        for (int i = 0; i < pythonParameters.length; i++) {
            pythonParams[i] = pythonParameters[i] + " : " + toPythonParam(parameterTypes[i]);
        }

        return String.join(", ", pythonParams);
    }

    private static String toPythonParam(Class<?> parameterType) {
        if (Vertex.class.isAssignableFrom(parameterType) || Tensor.class.isAssignableFrom(parameterType)) {
            return "mypy_vertex_arg_types";
        } else if (parameterType.isArray()) {
            return "mypy_shape_types";
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

        PythonConstructor(String javaClass, String pythonClass, String pythonTypedParameters, String pythonParameters) {
            this.javaClass = javaClass;
            this.pythonClass = pythonClass;
            this.pythonTypedParameters = pythonTypedParameters;
            this.pythonParameters = pythonParameters;
        }
    }

}
