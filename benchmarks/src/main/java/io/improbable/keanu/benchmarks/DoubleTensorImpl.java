package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensorFactory;

public enum DoubleTensorImpl {
    JVM {
        @Override
        DoubleTensorFactory getFactory() {
            return new JVMDoubleTensorFactory();
        }
    },
    ND4J {
        @Override
        DoubleTensorFactory getFactory() {
            return new Nd4jDoubleTensorFactory();
        }
    };

    abstract DoubleTensorFactory getFactory();
}
