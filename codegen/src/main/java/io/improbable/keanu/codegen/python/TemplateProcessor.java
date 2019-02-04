package io.improbable.keanu.codegen.python;

import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateException;
import lombok.experimental.UtilityClass;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

@UtilityClass
public class TemplateProcessor {

    public void processDataModel(Map<String, Object> dataModel, Template fileTemplate, Writer fileWriter) {
        try {
            fileTemplate.process(dataModel, fileWriter);
            fileWriter.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (TemplateException e) {
            throw new RuntimeException(e);
        }
    }

    public Template getFileTemplate(String templateFile) {
        Configuration cfg = new Configuration(Configuration.VERSION_2_3_28);
        cfg.setClassForTemplateLoading(Runner.class, "/");
        try {
            return cfg.getTemplate(templateFile);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public Writer createFileWriter(String fileToWrite) {
        try {
            Files.deleteIfExists(Paths.get(fileToWrite));
            return new FileWriter(new File(fileToWrite), true);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}