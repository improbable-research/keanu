package io.improbable.keanu.util.csv.pojo;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.stream.Collectors;

import static org.apache.commons.lang3.ArrayUtils.toObject;

class CsvColumnDeserializer {

    private CsvColumnDeserializer() {
    }

    static Object convertToAppropriateType(List<String> s, Class<?> fieldType) {
        if (fieldType == int[].class) {
            return convertToIntegers(s);
        }

        if (fieldType == Integer[].class) {
            return toObject(convertToIntegers(s));
        }

        if (fieldType == IntegerTensor.class) {
            return IntegerTensor.create(convertToIntegers(s));
        }

        if (fieldType == double[].class) {
            return convertToDoubles(s);
        }

        if (fieldType == Double[].class) {
            return toObject(convertToDoubles(s));
        }

        if (fieldType == DoubleTensor.class) {
            return DoubleTensor.create(convertToDoubles(s));
        }

        if (fieldType == boolean[].class) {
            return convertToBooleans(s);
        }
        
        if (fieldType == Boolean[].class) {
            return toObject(convertToBooleans(s));
        }

        if (fieldType == BooleanTensor.class) {
            return BooleanTensor.create(convertToBooleans(s));
        }

        throw new IllegalArgumentException("Could not convert " + s + " to " + fieldType);
    }

    private static int[] convertToIntegers(List<String> data) {
        return data.stream()
            .mapToInt(Integer::parseInt)
            .toArray();
    }

    private static double[] convertToDoubles(List<String> data) {
        return data.stream()
            .mapToDouble(Double::parseDouble)
            .toArray();
    }

    /**
     * Accepts "true", "t" and any string number that parses to 1.0
     *
     * @param data list of strings to convertt
     * @return boolean array
     */
    private static boolean[] convertToBooleans(List<String> data) {

        List<Boolean> bools = data.stream().map(val -> {
            if (StringUtils.isNumeric(val)) {
                return Double.parseDouble(val) == 1.0;
            } else {
                return val.equalsIgnoreCase("true") || val.equalsIgnoreCase("t");
            }
        }).collect(Collectors.toList());

        return ArrayUtils.toPrimitive(bools.toArray(new Boolean[data.size()]));
    }
}
