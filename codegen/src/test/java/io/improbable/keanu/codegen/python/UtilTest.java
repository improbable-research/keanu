package io.improbable.keanu.codegen.python;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class UtilTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Rule
    public TemporaryFolder testFolder = new TemporaryFolder();

    private static String TEST_GENERATED_FILE = "test.py";
    private static String TEST_TEMPLATE_FILE = "test.py.ftl";

    @Test
    public void canCreateFileWriterForNewFile() {
        Path path = Paths.get(testFolder.getRoot().toString(), TEST_GENERATED_FILE);

        assertFalse(Files.exists(path));

        Util.createFileWriter(path.toAbsolutePath().toString());
        assertTrue(Files.exists(path));
    }

    @Test
    public void canCreateFileWriterForExistingFile() throws IOException {
        Path path = Paths.get(testFolder.getRoot().toString(), TEST_GENERATED_FILE);

        testFolder.newFile(TEST_GENERATED_FILE);
        assertTrue(Files.exists(path));

        Util.
            gcreateFileWriter(path.toAbsolutePath().toString());
        assertTrue(Files.exists(path));
    }

    @Test
    public void canCreateTemplate() {
        Util.getFileTemplate(TEST_TEMPLATE_FILE);
    }
}
