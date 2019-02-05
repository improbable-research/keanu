package io.improbable.keanu.codegen.python;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.sun.javadoc.ClassDoc;
import com.sun.javadoc.ConstructorDoc;
import com.sun.javadoc.RootDoc;
import com.sun.tools.doclets.standard.Standard;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import lombok.experimental.UtilityClass;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Reader;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@UtilityClass
public class KeanuProjectDoclet extends Standard {

    private final Logger logger = LoggerFactory.getLogger(KeanuProjectDoclet.class);

    private final String EXPORT_VERTEX_ANNOTATION_NAME = "@" + ExportVertexToPythonBindings.class.getName();
    private final String DESTINATION_FILE_NAME = "javadocstrings.json";
    private final String DESTINATION = "./build/resources/";

    public boolean start(RootDoc root) {

        Map<String, DocString> docStrings = new HashMap<>();
        ClassDoc[] classes = root.classes();
        for (ClassDoc classDoc : classes) {
            for (ConstructorDoc constructorDoc : classDoc.constructors()) {
                if (constructorDoc.annotations().length != 0) {
                    if (isConstructorAnnotated(constructorDoc)) {
                        Map<String, String> params = ParamStringProcessor.getNameToCommentMapping(constructorDoc);
                        DocString docString = new DocString(constructorDoc.commentText(), params);
                        docStrings.put(constructorDoc.qualifiedName(), docString);
                    }
                }
            }
        }
        writeDocStringsToFile(docStrings);
        return true;
    }

    private boolean isConstructorAnnotated(ConstructorDoc constructorDoc) {
        return Arrays.stream(constructorDoc.annotations())
            .anyMatch(an -> an.toString().equals(EXPORT_VERTEX_ANNOTATION_NAME));
    }

    private void writeDocStringsToFile(Map<String, DocString> docString) {
        String json = (new Gson()).toJson(docString);
        createFilesIfNecessary();
        try (OutputStream outputStream =
                 new FileOutputStream(DESTINATION + DESTINATION_FILE_NAME, false)) {
            outputStream.write(json.getBytes());
        } catch (IOException e) {
            logger.error("Could not write to file while processing JavaDoc strings", e);
        }
    }

    private void createFilesIfNecessary() {
        File outputFile = new File(DESTINATION + DESTINATION_FILE_NAME);
        outputFile.getParentFile().mkdirs();
        try {
            outputFile.createNewFile();
        } catch (IOException e) {
            logger.error("Could not write to file while processing JavaDoc strings", e);
        }
    }

    public Map<String, DocString> getDocStringsFromFile() throws IOException {
        try {
            Type listType = new TypeToken<Map<String, DocString>>(){}.getType();
            Reader reader = new FileReader(DESTINATION + DESTINATION_FILE_NAME);
            return (new Gson()).fromJson(reader, listType);
        } catch (IOException e) {
            throw new IOException("Could not read JavaDoc strings from file at " + System.getProperty("user.dir") + DESTINATION + DESTINATION_FILE_NAME, e);
        }
    }
}
