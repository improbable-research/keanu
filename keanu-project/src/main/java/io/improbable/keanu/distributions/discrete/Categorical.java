package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Categorical<CATEGORY> implements Distribution<GenericTensor<CATEGORY>> {

    private final Map<CATEGORY, DoubleTensor> selectableValues;
    private final List<CATEGORY> categoryOrder;

    public static <CAT> Categorical<CAT> withParameters(Map<CAT, DoubleTensor> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<CATEGORY, DoubleTensor> selectableValues) {
        this.selectableValues = new LinkedHashMap<>(selectableValues);
        this.categoryOrder = new ArrayList<>(this.selectableValues.keySet());
    }

    public GenericTensor<CATEGORY> sample(long[] shape, KeanuRandom random) {

        DoubleTensor sumOfProbabilities = getSumOfProbabilities(shape);

        DoubleTensor p = random.nextDouble(shape);
        DoubleTensor sum = DoubleTensor.zeros(shape);

        CATEGORY lastValue = categoryOrder.get(categoryOrder.size() - 1);
        GenericTensor<CATEGORY> sample = GenericTensor.createFilled(lastValue, shape);
        BooleanTensor sampleValuesSetSoFar = BooleanTensor.falses(shape);

        for (CATEGORY category : categoryOrder) {
            DoubleTensor probabilitiesForCategory = selectableValues.get(category);

            DoubleTensor normalizedProbabilities = probabilitiesForCategory.div(sumOfProbabilities);
            sum = sum.plus(normalizedProbabilities);

            BooleanTensor maskForUnassignedSampleValues = sampleValuesSetSoFar.xor(sum.greaterThan(p));

            sample = GenericTensor.scalar(category).where(maskForUnassignedSampleValues, sample);

            sampleValuesSetSoFar.orInPlace(maskForUnassignedSampleValues);

            if (sampleValuesSetSoFar.allTrue().scalar()) {
                break;
            }
        }

        return sample;
    }

    public DoubleTensor logProb(GenericTensor<CATEGORY> x) {

        DoubleTensor sumOfProbabilities = getSumOfProbabilities(x.getShape());

        DoubleTensor logProb = DoubleTensor.zeros(x.getShape());
        for (Map.Entry<CATEGORY, DoubleTensor> entry : selectableValues.entrySet()) {

            DoubleTensor xEqualToEntryKeyMask = x.elementwiseEquals(GenericTensor.createFilled(entry.getKey(), x.getShape())).toDoubleMask();
            logProb = logProb.plus(xEqualToEntryKeyMask.timesInPlace(entry.getValue().div(sumOfProbabilities).logInPlace()));
        }

        return logProb;
    }

    private boolean containsNonPositiveEntry(DoubleTensor sumOfProbabilities) {
        return !sumOfProbabilities.lessThanOrEqual(0.).allFalse().scalar();
    }

    private DoubleTensor getSumOfProbabilities(long[] shape) {

        DoubleTensor sumOfProbabilities = DoubleTensor.zeros(shape);
        for (DoubleTensor p : selectableValues.values()) {
            sumOfProbabilities = sumOfProbabilities.plus(p);
        }

        if (containsNonPositiveEntry(sumOfProbabilities)) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        return sumOfProbabilities;
    }
}
