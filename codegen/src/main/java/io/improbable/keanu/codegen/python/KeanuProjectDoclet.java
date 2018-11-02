package io.improbable.keanu.codegen.python;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.sun.javadoc.*;
import com.sun.tools.doclets.standard.Standard;

import java.lang.reflect.Type;
import java.io.*;
import java.util.*;

public class KeanuProjectDoclet extends Standard {

    private static final String JAVADOC_STRING_FILE_NAME = "javadocstrings.txt";

    public static boolean start(RootDoc root) {

        Map<String, DocString> docStrings = new HashMap<>();
        ClassDoc[] classes = root.classes();
        for (ClassDoc classDoc : classes) {
            for (ConstructorDoc constructorDoc : classDoc.constructors()) {
                if (constructorDoc.annotations().length != 0) {
                    if (isConstructorAnnotated(constructorDoc)) {
                        Map<String, String> params = ParamStringProcessor.getNameToCommentMapping(constructorDoc);
                        DocString docString = new DocString(constructorDoc.commentText(), params, constructorDoc.qualifiedName());
                        docStrings.put(constructorDoc.qualifiedName(), docString);
                    }
                }
            }
        }
        writeDocStringsToFile(docStrings);
        return true;
    }

    private static boolean isConstructorAnnotated(ConstructorDoc constructorDoc) {
        for (AnnotationDesc an: constructorDoc.annotations()) {
            if (an.toString().contains("ExportVertexToPythonBindings")) {
                return true;
            }
        }
        return false;
    }

    private static void writeDocStringsToFile(Map<String, DocString> docString) {
        try {
            Gson gson = new Gson();
            String json = gson.toJson(docString);
            OutputStream outputStream = new FileOutputStream("./" + JAVADOC_STRING_FILE_NAME);
            System.out.println("Writing docstrings to " + System.getProperty("user.dir") + "/" + JAVADOC_STRING_FILE_NAME);
            outputStream.write(json.getBytes());
            System.out.println("Finished writing docstrings to " + System.getProperty("user.dir") + "/" + JAVADOC_STRING_FILE_NAME);
        } catch (IOException e) {
            System.out.println("Could not write to file while processing JavaDoc strings");
        }
    }

    public static Map<String, DocString> getDocStringsFromFile() {
        try {
            Type listType = new TypeToken<Map<String, DocString>>(){}.getType();
            Gson gson = new Gson();
            Reader reader = new FileReader("./" + JAVADOC_STRING_FILE_NAME);
            return gson.fromJson(reader, listType);
        } catch (IOException e) {
            System.out.println("Could not read JavaDoc strings from file");
            return new HashMap<>();
        }
    }
}

