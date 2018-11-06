package io.improbable.keanu.util.dot;

import org.apache.commons.io.FilenameUtils;

import java.io.File;

/**
 * Utility class for creating a unique file in the given location. Used by Dot output writer.
 */
public class DotWriterUtils {

    // Make sure the output directory exists and that no files are being overwritten.
    protected static File getOutputFile(String fileName) {

        File outputFile = getFreshOutputFile(fileName);

        // Make sure the output directory exists.
        if (outputFile.getParentFile() != null && !outputFile.getParentFile().exists()){
            outputFile.getParentFile().mkdirs();
        }

        return outputFile;
    }

    // Add an index to the filename if a file with the specified name already exists.
    private static File getFreshOutputFile(String fileName) {
        File outputFile = new File(fileName);

        if (outputFile.exists()) {
            String baseName = FilenameUtils.getBaseName(fileName);
            String extension = FilenameUtils.getExtension(fileName);
            int counter = 1;
            while(outputFile.exists()) {
                outputFile = new File(outputFile.getParent(), baseName + "_" + (counter++) + "." + extension);
            }
        }

        return outputFile;
    }
}
