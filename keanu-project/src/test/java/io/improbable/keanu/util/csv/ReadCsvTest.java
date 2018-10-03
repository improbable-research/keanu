package io.improbable.keanu.util.csv;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;

public class ReadCsvTest {

  String csv = "a,b,c\nhel,lo,world";

  @Test
  public void canReadFromString() {
    canRead(ReadCsv.fromString(csv));
  }

  @Test
  public void canReadFromResource() {
    canRead(ReadCsv.fromResources("data/simple_data.csv"));
  }

  @Test
  public void canReadFromFile() throws IOException {
    Path temp = Files.createTempFile("simple_data", ".csv");
    Files.write(temp, csv.getBytes());
    temp.toFile().deleteOnExit();

    CsvReader csvReader = ReadCsv.fromFile(temp);
    canRead(csvReader);
  }

  private void canRead(CsvReader csvReader) {
    List<List<String>> lines = csvReader.readLines();
    assertTrue(lines.size() == 1);
    assertTrue(csvReader.getHeader().equals(Arrays.asList("a", "b", "c")));
    assertTrue(lines.get(0).equals(Arrays.asList("hel", "lo", "world")));
  }
}
