package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import io.improbable.keanu.tensor.lng.JVMLongTensorFactory;
import io.improbable.keanu.tensor.lng.LongTensorFactory;

public class TensorFactories {

    public static DoubleTensorFactory doubleTensorFactory = new JVMDoubleTensorFactory();
    public static LongTensorFactory longTensorFactory = new JVMLongTensorFactory();
}
