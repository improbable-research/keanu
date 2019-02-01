package io.improbable.keanu.util.io;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.mir.MIR;
import io.improbable.mir.SavedBayesNet;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MIRSaver extends ProtobufSaver {

    final static String DEFAULT_MODEL_NAME = "Saved Keanu Graph";
    final static String ENTRY_POINT_NAME = "Keanu Graph";

    private final String modelName;

    public MIRSaver(BayesianNetwork net) {
        this(net, DEFAULT_MODEL_NAME);
    }

    public MIRSaver(BayesianNetwork net, String modelName) {
        super(net);
        this.modelName = modelName;
    }

    @Override
    public void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException {
        SavedBayesNet.Graph graph = getGraph(saveValues);
        MIR.Model myModel = wrapGraphInMIR(graph, metadata);
        myModel.writeTo(output);
        clearGraph();
    }

    private MIR.Model wrapGraphInMIR(SavedBayesNet.Graph graph, Map<String, String> metadata) {
        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setName(modelName);
        builder.setEntryPointName(ENTRY_POINT_NAME);
        builder.setProperties(getBasicModelProperties(metadata));
        builder.putAllFunctionsByName(getFunctionMap(graph));

        return builder.build();
    }

    private MIR.ModelProperties getBasicModelProperties(Map<String, String> metadata) {
        MIR.CycleMetadata cycles = MIR.CycleMetadata.newBuilder()
            .setDimensionGenerating(false)
            .setIteration(MIR.IterationType.NONE)
            .build();

        return MIR.ModelProperties.newBuilder()
            .setMirVersion(MIR.VersionNumber.VERSION_1)
            .setLoopMetadata(cycles)
            .setRecursionMetadata(cycles)
            .setDynamicCollections(false)
            .addAllMetadata(convertMetadata(metadata))
            .build();
    }

    private List<SavedBayesNet.Metadata> convertMetadata(Map<String, String> metadata) {
        if (metadata == null) {
            return new ArrayList<>();
        } else {
            return metadata.entrySet().stream()
                .map(entry ->
                    SavedBayesNet.Metadata.newBuilder().setKey(entry.getKey()).setValue(entry.getValue()).build())
                .collect(Collectors.toList());
        }
    }

    private Map<String, MIR.Function> getFunctionMap(SavedBayesNet.Graph graph) {
        Map<String, MIR.Function> functionMap = new HashMap<>();
        functionMap.put(ENTRY_POINT_NAME, getFunctionForGraph(graph));

        return functionMap;
    }

    private MIR.Function getFunctionForGraph(SavedBayesNet.Graph graph) {
        MIR.Function.Builder builder = MIR.Function.newBuilder();

        builder.setName(ENTRY_POINT_NAME);
        builder.addInstructionGroupsBuilder(0)
            .setId(0)
            .setGraph(graph);

        return builder.build();
    }
}
