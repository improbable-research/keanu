package io.improbable.keanu.util.csv.pojo;

import io.improbable.keanu.util.csv.pojo.bycolumn.CsvColumnConsumer;
import io.improbable.keanu.util.csv.pojo.byrow.CsvCellConsumer;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;

/**
 * This finds an appropriate setter method for a given csv title. This is done by case insensitive name
 * or by annotation.
 */
public class SetterMatcher {

    private SetterMatcher() {
    }

    public static <T> Optional<CsvCellConsumer<T>> getSetterCellConsumer(String title, List<Method> potentialMethods) {

        final Optional<Method> matchingMethodMaybe = findMatchingSetter(title.trim(), potentialMethods);

        return matchingMethodMaybe
            .map(SetterMatcher::createCellConsumerForMethod);
    }

    private static <T> CsvCellConsumer<T> createCellConsumerForMethod(Method method) {

        return (target, value) -> {

            Object convertedValue = CsvCellDeserializer.convertToAppropriateType(value, method.getParameterTypes()[0]);

            try {
                method.invoke(target, convertedValue);
            } catch (InvocationTargetException | IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        };
    }

    public static <T> Optional<CsvColumnConsumer<T>> getSetterColumnConsumer(String title, List<Method> potentialMethods) {

        final Optional<Method> matchingMethodMaybe = findMatchingSetter(title.trim(), potentialMethods);

        return matchingMethodMaybe
            .map(SetterMatcher::createColumnConsumerForMethod);
    }

    private static <T> CsvColumnConsumer<T> createColumnConsumerForMethod(Method method) {
        return (target, value) -> {

            Object convertedValue = CsvColumnDeserializer.convertToAppropriateType(value, method.getParameterTypes()[0]);

            try {
                method.invoke(target, convertedValue);
            } catch (InvocationTargetException | IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        };
    }

    /**
     * @param title      title to match to
     * @param potentials list of all methods on target class
     * @return A method that either matches by name prepended with set (e.g. myValue would match setMyValue)
     * or has a CsvProperty annotation with an EXACT name match.
     */
    private static Optional<Method> findMatchingSetter(final String title, final List<Method> potentials) {
        return potentials.stream()
            .filter(SetterMatcher::methodTakesOnlyOneArgument)
            .filter(method -> titleMatchesSetterMethod(method, title) || methodContainsPropertyAnnotation(method, title))
            .findFirst();
    }

    private static boolean methodTakesOnlyOneArgument(Method method) {
        return method.getParameterTypes().length == 1;
    }

    private static boolean titleMatchesSetterMethod(Method method, String title) {
        return method.getName().equalsIgnoreCase("set" + title);
    }

    private static boolean methodContainsPropertyAnnotation(Method method, String title) {

        if (method.isAnnotationPresent(CsvProperty.class)) {
            CsvProperty annotation = method.getAnnotation(CsvProperty.class);
            return annotation.value().equals(title);
        }

        return false;
    }
}
