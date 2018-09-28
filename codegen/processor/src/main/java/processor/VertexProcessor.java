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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

class VertexProcessor {

    static void process(String dir) throws IOException, TemplateException {
        File file = new File(dir + "vertex.py");
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
        Template vertexTemplate = cfg.getTemplate("vertex.py.ftl");

        Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner()));

        Set<Constructor> constructors = reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class);

        Map<String, Object> input = new HashMap<>();
        input.put("size", constructors.size());
        int index = 1;

        for (Constructor constructor : constructors) {
            String str = String.valueOf(index);
            String javaKlass = constructor.getDeclaringClass().getSimpleName();

            input.put("package" + str, constructor.getDeclaringClass().getCanonicalName());
            input.put("klass" + str, javaKlass);
            input.put("py_klass" + str, javaKlass.replaceAll("Vertex$", ""));

            index++;
        }
        vertexTemplate.process(input, fileWriter);
    }

}
