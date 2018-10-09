package io.improbable.keanu.model.linear;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ClassificationModel extends LinearModel {

    default double accuracy(DoubleTensor x, BooleanTensor yTrue) {
        BooleanTensor yPredicted = predict(x).greaterThan(0.5);
        return yPredicted.elementwiseEquals(yTrue).toDoubleMask().sum() / yTrue.getLength();
    }

}
