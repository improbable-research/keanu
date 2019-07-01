package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;

public class TensorFactories {

    public static DoubleTensorFactory doubleTensorFactory = new JVMDoubleTensorFactory();
//    public static DoubleTensorFactory doubleTensorFactory = new Nd4jDoubleTensorFactory();

}
