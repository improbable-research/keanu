package processor;

import annotation.ExportVertexToPythonBindings;
import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateException;
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
        File file = new File(dir + GENERATED_FILE);
        if (file.exists() && (!file.delete() || !file.createNewFile())) {
            throw new FileNotFoundException("Couldn't regenerate file: " + file.getPath());
        }
        Writer fileWriter = new FileWriter(file, true);

        Configuration cfg = new Configuration(Configuration.VERSION_2_3_28);
        cfg.setClassForTemplateLoading(Runner.class, "/");

        generateVertices(cfg, fileWriter);

        fileWriter.close();
    }

    private static void generateVertices(Configuration cfg, Writer fileWriter) throws IOException, TemplateException {
        Template vertexTemplate = cfg.getTemplate(TEMPLATE_FILE);

        Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner()));

        List<Constructor> constructors = new ArrayList<>(reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class));
        constructors.sort(Comparator.comparing(Constructor::getName));

        Map<String, Object> input = new HashMap<>();
        input.put("size", constructors.size());
        int index = 1;

        for (Constructor constructor : constructors) {
            String javaClass = constructor.getDeclaringClass().getSimpleName();

            input.put("package" + index, constructor.getDeclaringClass().getCanonicalName());
            input.put("class" + index, javaClass);
            input.put("py_class" + index, toPythonClass(javaClass));

            index++;
        }
        vertexTemplate.process(input, fileWriter);
    }

    private static String toPythonClass(String javaClass) {
        return javaClass.replaceAll("Vertex$", "");
    }
}
