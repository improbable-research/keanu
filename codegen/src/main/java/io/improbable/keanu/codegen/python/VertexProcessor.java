package io.improbable.keanu.codegen.python;

import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateException;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import lombok.Getter;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.scanners.TypeAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
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

    static void process(String dir) throws IOException, TemplateException {
        Configuration cfg = new Configuration(Configuration.VERSION_2_3_28);
        cfg.setClassForTemplateLoading(Runner.class, "/");
        Template fileTemplate = cfg.getTemplate(TEMPLATE_FILE);

        File file = new File(dir + GENERATED_FILE);
        if (file.exists() && (!file.delete() || !file.createNewFile())) {
            throw new FileNotFoundException("Couldn't regenerate file: " + file.getPath());
        }
        Writer fileWriter = new FileWriter(file, true);

        generateFile(fileTemplate, fileWriter);

        fileWriter.close();
    }

    private static void generateFile(Template fileTemplate, Writer fileWriter) throws IOException, TemplateException {
        List<Constructor> constructors = getSortedListOfAnnotatedVertexConstructors();
        Map<String, Object> dataModel = buildDataModel(constructors);
        fileTemplate.process(dataModel, fileWriter);
    }

    private static List<Constructor> getSortedListOfAnnotatedVertexConstructors() {
         Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner()));

        List<Constructor> constructors = new ArrayList<>(reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class));
        constructors.sort(Comparator.comparing(Constructor::getName));

        return constructors;
    }

    private static Map<String, Object> buildDataModel(List<Constructor> constructors) {
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
