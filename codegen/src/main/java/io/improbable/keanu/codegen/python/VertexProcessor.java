package io.improbable.keanu.codegen.python;

import com.google.common.base.CaseFormat;
import freemarker.template.Template;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import lombok.Getter;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.io.Writer;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class VertexProcessor {

    final private static String TEMPLATE_FILE = "vertex.py.ftl";
    final private static String GENERATED_FILE = "vertex.py";

    static void process(String generatedDir) {
        Map<String, Object> dataModel = buildDataModel();
        Template fileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_FILE);
        Writer fileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_FILE);

        TemplateProcessor.processDataModel(dataModel, fileTemplate, fileWriter);
    }

    private static Map<String, Object> buildDataModel() {
        List<Constructor> constructors = getSortedListOfAnnotatedVertexConstructors();

        Map<String, Object> root = new HashMap<>();
        List<Import> imports = new ArrayList<>();
        List<PythonConstructor> pythonConstructors = new ArrayList<>();
        root.put("imports", imports);
        root.put("constructors", pythonConstructors);

        for (Constructor constructor : constructors) {
            String javaClass = constructor.getDeclaringClass().getSimpleName();

            String[] pythonParameters = Arrays.stream(constructor.getParameters()).map(
                parameter -> CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, parameter.getName())).toArray(String[]::new);

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));
            pythonConstructors.add(new PythonConstructor(javaClass, toPythonClass(javaClass), String.join(", ", pythonParameters)));
        }

        return root;
    }

    private static List<Constructor> getSortedListOfAnnotatedVertexConstructors() {
         Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner()));

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
        private String pythonParameters;

        PythonConstructor(String javaClass, String pythonClass, String pythonParameters) {
            this.javaClass = javaClass;
            this.pythonClass = pythonClass;
            this.pythonParameters = pythonParameters;
        }
    }

}
