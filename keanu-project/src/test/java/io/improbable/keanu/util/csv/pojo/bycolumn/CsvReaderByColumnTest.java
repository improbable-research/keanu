package io.improbable.keanu.util.csv.pojo.bycolumn;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.util.csv.pojo.CsvProperty;
import org.junit.Test;

import static org.apache.commons.lang3.ArrayUtils.toPrimitive;
import static org.junit.Assert.assertArrayEquals;

public class CsvReaderByColumnTest {

    String csv = "a,b,c\n" +
        "0,2.5,t\n" +
        "1,3,true\n" +
        "-3,4e3,0";

    @Test
    public void givenCsvStringThenLoadAsPOJO() {
        TestPOJO testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestPOJO.class)
            .load();

        assertCorrectlyDeserialized(toPrimitive(testPOJO.a), testPOJO.b, testPOJO.c);
    }

    @Test
    public void givenCsvStringThenLoadAsPOJOWithSetters() {
        TestPOJOWithSetters testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestPOJOWithSetters.class)
            .load();

        assertCorrectlyDeserialized(testPOJO.a, toPrimitive(testPOJO.b), testPOJO.c);
    }

    @Test
    public void givenCsvStringThenLoadAsPOJOWithAnnotations() {
        TestPOJOWithAnnotations testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestPOJOWithAnnotations.class)
            .load();

        assertCorrectlyDeserialized(testPOJO.a, testPOJO.b, toPrimitive(testPOJO.c));
    }

    @Test
    public void givenCsvStringThenLoadAsTensorPOJO() {
        TestTensorPOJO testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestTensorPOJO.class)
            .load();

        assertCorrectlyDeserialized(testPOJO.a.asFlatIntegerArray(), testPOJO.b.asFlatDoubleArray(), toPrimitive(testPOJO.c.asFlatArray()));
    }

    @Test
    public void givenCsvStringThenLoadAsTensorPOJOWithSetters() {
        TestTensorPOJOWithSetters testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestTensorPOJOWithSetters.class)
            .load();

        assertCorrectlyDeserialized(testPOJO.a.asFlatIntegerArray(), testPOJO.b.asFlatDoubleArray(), toPrimitive(testPOJO.c.asFlatArray()));
    }

    @Test
    public void givenCsvStringThenLoadAsTensorPOJOWithAnnotations() {
        TestTensorPOJOWithAnnotations testPOJO = ReadCsv.fromString(csv)
            .asVectorizedColumnsDefinedBy(TestTensorPOJOWithAnnotations.class)
            .load();

        assertCorrectlyDeserialized(testPOJO.a.asFlatIntegerArray(), testPOJO.b.asFlatDoubleArray(), toPrimitive(testPOJO.c.asFlatArray()));
    }

    private void assertCorrectlyDeserialized(int[] actualA, double[] actualB, boolean[] actualC) {
        assertArrayEquals(new int[]{0, 1, -3}, actualA);
        assertArrayEquals(new double[]{2.5, 3.0, 4e3}, actualB, 0.0);
        assertArrayEquals(new boolean[]{true, true, false}, actualC);
    }

    public static class TestPOJO {
        public Integer[] a;
        public double[] b;
        public boolean[] c;
    }

    public static class TestPOJOWithSetters {
        private int[] a;
        private Double[] b;
        private boolean[] c;

        public void setA(int[] a) {
            this.a = a;
        }

        public void setB(Double[] b) {
            this.b = b;
        }

        public void setC(boolean[] c) {
            this.c = c;
        }
    }

    public static class TestPOJOWithAnnotations {
        private int[] a;
        private double[] b;
        private Boolean[] c;

        @CsvProperty("a")
        public void setSomeThing(int[] a) {
            this.a = a;
        }

        @CsvProperty("b")
        public void setSomethingElse(double[] b) {
            this.b = b;
        }

        @CsvProperty("c")
        public void setSomeBooleans(Boolean[] c) {
            this.c = c;
        }
    }

    public static class TestTensorPOJO {
        public IntegerTensor a;
        public DoubleTensor b;
        public BooleanTensor c;
    }

    public static class TestTensorPOJOWithSetters {
        private IntegerTensor a;
        private DoubleTensor b;
        private BooleanTensor c;

        public void setA(IntegerTensor a) {
            this.a = a;
        }

        public void setB(DoubleTensor b) {
            this.b = b;
        }

        public void setC(BooleanTensor c) {
            this.c = c;
        }
    }

    public static class TestTensorPOJOWithAnnotations {
        private IntegerTensor a;
        private DoubleTensor b;
        private BooleanTensor c;

        @CsvProperty("a")
        public void setSomeThing(IntegerTensor a) {
            this.a = a;
        }

        @CsvProperty("b")
        public void setSomethingElse(DoubleTensor b) {
            this.b = b;
        }

        @CsvProperty("c")
        public void setSomeBooleans(BooleanTensor c) {
            this.c = c;
        }
    }
}
