package io.improbable.keanu.backend.tensorflow;

import static java.util.Collections.singletonList;

import static io.improbable.keanu.backend.tensorflow.TensorflowData.toBooleanTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toDoubleTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toIntegerTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toTensorFlow;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import lombok.Getter;

public class TensorflowComputableGraph implements ComputableGraph {

    private final Session session;

    @Getter
    private final Scope scope;

    @Getter
    private final GraphBuilder graphBuilder;

    public TensorflowComputableGraph(Session session, Scope scope) {
        this.session = session;
        this.scope = scope;
        this.graphBuilder = new GraphBuilder(scope);
    }

    @Override
    public Map<String, ?> eval(Map<String, ?> inputs, Collection<String> outputs) {

        Session.Runner runner = feedInputs(inputs);
        List<Tensor<?>> tfResults = fetchOutputs(runner, outputs).run();

        Map<String, ?> results = new HashMap<>();
        Iterator<Tensor<?>> resultIterator = tfResults.iterator();
        for (String output: outputs) {
            results.put(output, convertToKeanuTensor(resultIterator.next()));
        }

        return results;
    }

    public <T> T eval(Map<String, ?> inputs, String output) {
        Session.Runner runner = feedInputs(inputs);
        List<Tensor<?>> tfResults = fetchOutputs(runner, singletonList(output)).run();
        return convertToKeanuTensor(tfResults.get(0));
    }

    private <T> T convertToKeanuTensor(Tensor<?> tensor) {
        try (Tensor<?> tfResult = tensor) {
            switch (tfResult.dataType()) {
                case DOUBLE:
                    return (T) toDoubleTensor(tfResult);
                case BOOL:
                    return (T) toBooleanTensor(tfResult);
                case INT32:
                    return (T) toIntegerTensor(tfResult);
                default:
                    throw new IllegalArgumentException("Cannot fetch output of type " + tfResult.dataType());
            }
        }
    }

    private Session.Runner feedInputs(Map<String, ?> inputs) {
        Session.Runner runner = session.runner();
        for (Map.Entry<String, ?> inputEntry : inputs.entrySet()) {

            Object tensor = inputEntry.getValue();

            Tensor<?> tensorFlowTensor = null;
            if (tensor instanceof DoubleTensor) {
                tensorFlowTensor = toTensorFlow((DoubleTensor) tensor);
            } else if (tensor instanceof BooleanTensor) {
                tensorFlowTensor = toTensorFlow((BooleanTensor) tensor);
            } else if (tensor instanceof IntegerTensor) {
                tensorFlowTensor = toTensorFlow((IntegerTensor) tensor);
            }

            runner = runner.feed(inputEntry.getKey(), tensorFlowTensor);
        }

        return runner;
    }

    private Session.Runner fetchOutputs(Session.Runner runner, Collection<String> outputs) {

        Session.Runner fetchedRunner = runner;
        for (String output : outputs) {
            fetchedRunner = fetchedRunner.fetch(output);
        }

        return fetchedRunner;
    }

    @Override
    public void close() {
        session.close();
    }
}
