package io.improbable.keanu.codegen.python;

import freemarker.template.Template;
import static junit.framework.TestCase.assertTrue;
import lombok.Getter;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.Writer;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FreeMarkerTemplateProcessorTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Rule
    public TemporaryFolder testFolder = new TemporaryFolder();

    private static String TEST_GENERATED_FILE = "test.txt";
    private static String TEST_TEMPLATE_FILE = "test.txt.ftl";

    @Test
    public void canCreateFileWriterForNewFile() {
        Path path = Paths.get(testFolder.getRoot().toString(), TEST_GENERATED_FILE);

        assertFalse(Files.exists(path));

        FreeMarkerTemplateProcessor.createFileWriter(path.toAbsolutePath().toString());
        assertTrue(Files.exists(path));
    }

    @Test
    public void canCreateFileWriterForExistingFile() throws IOException {
        Path path = Paths.get(testFolder.getRoot().toString(), TEST_GENERATED_FILE);

        testFolder.newFile(TEST_GENERATED_FILE);
        assertTrue(Files.exists(path));

        FreeMarkerTemplateProcessor.createFileWriter(path.toAbsolutePath().toString());
        assertTrue(Files.exists(path));
    }

    @Test
    public void canCreateTemplate() {
        FreeMarkerTemplateProcessor.getFileTemplate(TEST_TEMPLATE_FILE);
    }


    @Test
    public void canProcessTemplate() throws IOException {
        Path generatedFilePath = Paths.get(testFolder.getRoot().toString(), TEST_GENERATED_FILE);
        Path expectedContentFilePath = Paths.get(ClassLoader.getSystemResource("result.txt").getPath());

        Writer writer = FreeMarkerTemplateProcessor.createFileWriter(generatedFilePath.toAbsolutePath().toString());
        Template template = FreeMarkerTemplateProcessor.getFileTemplate(TEST_TEMPLATE_FILE);
        Map<String, Object> dataModel = buildTestDataModel();

        FreeMarkerTemplateProcessor.processDataModel(dataModel, template, writer);

        assertFilesContainSameContent(generatedFilePath, expectedContentFilePath);
    }

    private Map<String, Object> buildTestDataModel() {
        List<Product> products = Arrays.asList(new Product("shoes"), new Product("pants"));
        int expenses = 100;
        Map<String, Object> dataModel = new HashMap<>();
        dataModel.put("products", products);
        dataModel.put("expenses", expenses);
        return dataModel;
    }

    private void assertFilesContainSameContent(Path generatedFilePath, Path expectedContentFilePath) throws IOException {
        byte[] generatedFileBytes = Files.readAllBytes(generatedFilePath);
        byte[] expectedContentFileBytes = Files.readAllBytes(expectedContentFilePath);

        assertArrayEquals(generatedFileBytes, expectedContentFileBytes);
    }

    public static class Product {
        @Getter
        String item;

        Product(String item) {
            this.item = item;
        }
    }
}
