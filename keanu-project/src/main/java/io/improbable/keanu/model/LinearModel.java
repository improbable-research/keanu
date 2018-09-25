package io.improbable.keanu.model;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface LinearModel extends Model {

    LinearModel fit();

    DoubleTensor predict(DoubleTensor x);

    default double score(DoubleTensor x, DoubleTensor yTrue) {
        DoubleTensor yPredicted = predict(x);
        double residualSumOfSquares = (yTrue.minus(yPredicted).pow(2.)).sum();
        double totalSumOfSquares = ((yTrue.minus(yTrue.average())).pow(2.)).sum();
        return 1 - (residualSumOfSquares / totalSumOfSquares);
    }

}
