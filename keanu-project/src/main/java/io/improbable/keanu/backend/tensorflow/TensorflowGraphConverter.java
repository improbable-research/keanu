package io.improbable.keanu.backend.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import io.improbable.keanu.network.BayesianNetwork;

public class TensorflowGraphConverter {


    public static void convert(BayesianNetwork network) {

        try (Graph g = new Graph()) {

            GraphBuilder b = new GraphBuilder(g);

            final Output<Double> result =
                b.div(
                    b.sub(
                        b.constant("a", 4.0),
                        b.constant("b", 2.0)
                    ),
                    b.constant("c", 6.0)
                );

            // Execute the "MyConst" operation in a Session.
            try (Session s = new Session(g);
                 // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
                 Tensor output = s.runner().fetch(result.op().name()).run().get(0)) {

                System.out.println(output.doubleValue());
            }
        }
    }


}
