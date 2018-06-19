package io.improbable.keanu.util.csv.pojo.byrow;

import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.util.csv.pojo.CsvProperty;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class CsvReaderByRowTest {

    String csv = "a,b,c\n" +
        "hel,lo,world";

    @Test
    public void givenCsvStringThenLoadAsPOJO() {
        List<TestPOJO> testPOJOS = ReadCsv.fromString(csv)
            .asRowsDefinedBy(TestPOJO.class)
            .load();

        TestPOJO actualPojo = testPOJOS.get(0);

        assertCorrectlyDeserialized(actualPojo.a, actualPojo.b, actualPojo.c);
    }

    @Test
    public void givenCsvStringThenLoadAsPOJOWithSetters() {
        List<TestPOJOWithSetter> testPOJOS = ReadCsv.fromString(csv)
            .asRowsDefinedBy(TestPOJOWithSetter.class)
            .load();

        TestPOJOWithSetter actualPojo = testPOJOS.get(0);

        assertCorrectlyDeserialized(actualPojo.a, actualPojo.b, actualPojo.c);
    }

    @Test
    public void givenCsvStringThenLoadAsPOJOWithAnnotations() {
        List<TestPOJOWithAnnotations> testPOJOS = ReadCsv.fromString(csv)
            .asRowsDefinedBy(TestPOJOWithAnnotations.class)
            .load();

        TestPOJOWithAnnotations actualPojo = testPOJOS.get(0);

        assertCorrectlyDeserialized(actualPojo.a, actualPojo.b, actualPojo.c);
    }

    private void assertCorrectlyDeserialized(String actualA, String actualB, String actualC) {
        assertEquals("hel", actualA);
        assertEquals("lo", actualB);
        assertEquals("world", actualC);
    }

    public static class TestPOJO {
        public String a;
        public String b;
        public String c;
    }

    public static class TestPOJOWithSetter {
        private String a;
        private String b;
        private String c;

        public void setA(String a) {
            this.a = a;
        }

        public void setB(String b) {
            this.b = b;
        }

        public void setC(String c) {
            this.c = c;
        }
    }

    public static class TestPOJOWithAnnotations {
        private String a;
        private String b;
        private String c;

        @CsvProperty("a")
        public void setSomething(String a) {
            this.a = a;
        }

        @CsvProperty("b")
        public void setSomethingElse(String b) {
            this.b = b;
        }

        @CsvProperty("c")
        public void setSomethingFinal(String c) {
            this.c = c;
        }
    }
}
