package io.improbable.keanu.model;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface LinearModel {

    LinearModel fit();

    DoubleTensor predict(DoubleTensor x);

}
