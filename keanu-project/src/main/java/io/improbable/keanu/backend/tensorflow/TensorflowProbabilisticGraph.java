package io.improbable.keanu.backend.tensorflow;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class TensorflowProbabilisticGraph implements ProbabilisticGraph {

    private final Session session;

    private final List<String> inputOrder;
    private final Map<String, Output<?>> gradientsByInput;

    @Override
    public double logProb(Map<String, DoubleTensor> inputs) {

        Session.Runner runner = feedInputs(inputs);

        try (Tensor result = runner.fetch(ProbabilisticGraph.LOG_PROB).run().get(0)) {
            return result.doubleValue();
        }
    }

    public List<DoubleTensor> getOutputs(Map<String, DoubleTensor> inputs, List<String> outputs) {
        Session.Runner runner = feedInputs(inputs);

        List<Tensor<?>> results = fetchOutputs(runner, outputs).run();

        List<DoubleTensor> resultTensors = new ArrayList<>();
        for (int i = 0; i < results.size(); i++) {
            Tensor<?> result = results.get(i);
            resultTensors.add(toDoubleTensor(result));
            result.close();
        }

        return resultTensors;
    }

    @Override
    public Map<String, DoubleTensor> logProbGradients(Map<String, DoubleTensor> inputs) {

        Session.Runner runner = feedInputs(inputs);

        List<String> gradientOutputOrder = inputOrder.stream()
            .map(input -> gradientsByInput.get(input).op().name())
            .collect(Collectors.toList());

        List<Tensor<?>> results = fetchOutputs(runner, gradientOutputOrder).run();

        Map<String, DoubleTensor> gradientResultsByInput = new HashMap<>();
        for (int i = 0; i < results.size(); i++) {
            Tensor<?> result = results.get(i);
            DoubleTensor gradient = toDoubleTensor(result);
            gradientResultsByInput.put(inputOrder.get(i), gradient);
            result.close();
        }

        return gradientResultsByInput;
    }

    private Session.Runner feedInputs(Map<String, DoubleTensor> inputs) {
        Session.Runner runner = session.runner();
        for (Map.Entry<String, DoubleTensor> inputEntry : inputs.entrySet()) {

            DoubleTensor tensor = inputEntry.getValue();

            Tensor<Double> input = Tensor.create(
                toLongs(tensor.getShape()),
                DoubleBuffer.wrap(tensor.asFlatDoubleArray())
            );

            runner = runner.feed(inputEntry.getKey(), input);
        }

        return runner;
    }

    private Session.Runner fetchOutputs(Session.Runner runner, List<String> outputs) {

        Session.Runner fetchedRunner = runner;
        for (String output : outputs) {
            fetchedRunner = fetchedRunner.fetch(output);
        }

        return fetchedRunner;
    }

    private DoubleTensor toDoubleTensor(Tensor<?> tensor) {

        DoubleBuffer buffer = DoubleBuffer.allocate(tensor.numElements());
        tensor.writeTo(buffer);
        double[] resultAsArray = buffer.array();

        return DoubleTensor.create(resultAsArray, toInts(tensor.shape()));
    }

    private static long[] toLongs(int[] ints) {
        long[] longs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) {
            longs[i] = ints[i];
        }
        return longs;
    }

    private static int[] toInts(long[] longs) {
        int[] ints = new int[longs.length];
        for (int i = 0; i < longs.length; i++) {
            ints[i] = (int) longs[i];
        }
        return ints;
    }

    @Override
    public void close() {
        session.close();
    }
}
