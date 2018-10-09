package io.improbable.keanu.codegen.python;

import freemarker.template.Template;
import freemarker.template.TemplateException;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import lombok.Getter;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class VertexProcessor {

    final private static String TEMPLATE_FILE = "vertex.py.ftl";
    final private static String GENERATED_FILE = "vertex.py";

    static void process(String generatedDir) {
        Map<String, Object> dataModel = buildDataModel();
        Writer fileWriter = FileUtil.createFileWriter(generatedDir + GENERATED_FILE);
        Template fileTemplate = FileUtil.getFileTemplate(TEMPLATE_FILE);

        try {
            fileTemplate.process(dataModel, fileWriter);
            fileWriter.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (TemplateException e) {
            throw new RuntimeException(e);
        }
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

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));
            pythonConstructors.add(new PythonConstructor(javaClass, toPythonClass(javaClass)));
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

        PythonConstructor(String javaClass, String pythonClass) {
            this.javaClass = javaClass;
            this.pythonClass = pythonClass;
        }
    }
}
