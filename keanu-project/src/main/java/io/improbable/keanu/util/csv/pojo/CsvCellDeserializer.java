package io.improbable.keanu.util.csv.pojo;

class CsvCellDeserializer {

    private CsvCellDeserializer() {
    }

    static Object convertToAppropriateType(String s, Class<?> fieldType) {
        if (fieldType == Integer.class || fieldType == Integer.TYPE) {
            return Integer.parseInt(s);
        }
        if (fieldType == Double.class || fieldType == Double.TYPE) {
            return Double.parseDouble(s);
        }
        if (fieldType == Float.class || fieldType == Float.TYPE) {
            return Float.parseFloat(s);
        }
        if (fieldType == Boolean.class || fieldType == Boolean.TYPE) {
            return Boolean.parseBoolean(s);
        }
        if (fieldType == String.class) {
            return s;
        }

        throw new IllegalArgumentException("Could not convert " + s + " to " + fieldType);
    }
}
