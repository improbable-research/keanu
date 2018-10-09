package io.improbable.keanu.codegen.python;

import freemarker.template.Configuration;
import freemarker.template.Template;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;

class FileUtil {
     static Writer createFileWriter(String fileToWrite) {
        File file = new File(fileToWrite);
        try {
            if (file.exists()) {
                file.delete();
                file.createNewFile();
            }
            return new FileWriter(file, true);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static Template getFileTemplate(String templateFile) {
        Configuration cfg = new Configuration(Configuration.VERSION_2_3_28);
        cfg.setClassForTemplateLoading(Runner.class, "/");
        try {
            return cfg.getTemplate(templateFile);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
