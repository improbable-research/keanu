package io.improbable.keanu.codegen.python;

import freemarker.template.Template;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.codegen.python.templateobject.Import;
import io.improbable.keanu.codegen.python.templateobject.VertexConstructor;
import lombok.experimental.UtilityClass;
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

@UtilityClass
class VertexProcessor {

    final private static String TEMPLATE_FILE = "generated.py.ftl";
    final private static String GENERATED_FILE = "generated.py";
    final private static String TEMPLATE_INIT_FILE = "__init__.py.ftl";
    final private static String GENERATED_INIT_FILE = "__init__.py";

    void process(String generatedDir) throws IOException {
        Map<String, Object> dataModel = buildDataModel();
        Template generatedVerticesFileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_FILE);
        Writer fileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_FILE);
        Template generatedInitFileTemplate = TemplateProcessor.getFileTemplate(TEMPLATE_INIT_FILE);
        Writer initFileWriter = TemplateProcessor.createFileWriter(generatedDir + GENERATED_INIT_FILE);

        TemplateProcessor.processDataModel(dataModel, generatedVerticesFileTemplate, fileWriter);
        TemplateProcessor.processDataModel(dataModel, generatedInitFileTemplate, initFileWriter);
    }

    private Map<String, Object> buildDataModel() throws IOException {
        Reflections reflections = new Reflections(new ConfigurationBuilder()
            .setUrls(ClasspathHelper.forPackage("io.improbable.keanu.vertices"))
            .setScanners(new MethodAnnotationsScanner(), new TypeAnnotationsScanner(), new MethodParameterNamesScanner()));

        List<Constructor> constructors = getSortedListOfAnnotatedVertexConstructors(reflections);

        Map<String, Object> root = new HashMap<>();
        List<Import> imports = new ArrayList<>();
        List<VertexConstructor> pythonConstructors = new ArrayList<>();

        root.put("imports", imports);
        root.put("constructors", pythonConstructors);
        Map<String, DocString> nameToDocStringMap = KeanuProjectDoclet.getDocStringsFromFile();
        StringJoiner exportedMethodsJoiner = new StringJoiner("\", \"", "\"", "\"");

        for (Constructor constructor : constructors) {
            String javaClass = constructor.getDeclaringClass().getSimpleName();
            String qualifiedName = constructor.getName();
            DocString docString = nameToDocStringMap.get(qualifiedName);
            PythonVertexConstructor pythonCtor = new PythonVertexConstructor(constructor, reflections);

            imports.add(new Import(constructor.getDeclaringClass().getCanonicalName()));

            VertexConstructor pythonConstructor = new VertexConstructor(
                javaClass,
                pythonCtor.getClassName(),
                pythonCtor.getChildClassName(),
                pythonCtor.getTypedParams(),
                pythonCtor.getCastedParams(),
                docString.getAsString()
            );

            pythonConstructors.add(pythonConstructor);
            exportedMethodsJoiner.add(pythonConstructor.getPythonClass());
        }
        root.put("exportedMethods", exportedMethodsJoiner.toString());

        return root;
    }

    private List<Constructor> getSortedListOfAnnotatedVertexConstructors(Reflections reflections) {
        return reflections.getConstructorsAnnotatedWith(ExportVertexToPythonBindings.class).stream()
            .sorted(Comparator.comparing(Constructor::getName))
            .collect(Collectors.toList());
    }

}