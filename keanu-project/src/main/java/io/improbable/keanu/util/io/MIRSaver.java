package io.improbable.keanu.util.io;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.mir.MIR;
import io.improbable.mir.SavedBayesNet;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

public class MIRSaver extends ProtobufSaver {

    public final static String MODEL_NAME = "Saved Keanu Graph";
    public final static String ENTRY_POINT_NAME = "Keanu Graph";

    public MIRSaver(BayesianNetwork net) {
        super(net);
    }

    @Override
    public void save(OutputStream output, boolean saveValues) throws IOException {
        SavedBayesNet.Graph graph = getGraph(saveValues);
        MIR.Model myModel = wrapGraphInMIR(graph);
        myModel.writeTo(output);
        clearGraph();
    }

    private MIR.Model wrapGraphInMIR(SavedBayesNet.Graph graph) {
        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setName(MODEL_NAME);
        builder.setEntryPointName(ENTRY_POINT_NAME);
        builder.setProperties(getBasicModelProperties());
        builder.putAllFunctionsByName(getFunctionMap(graph));

        return builder.build();
    }

    private MIR.ModelProperties getBasicModelProperties() {
        MIR.CycleMetadata cycles = MIR.CycleMetadata.newBuilder()
            .setDimensionGenerating(false)
            .setIteration(MIR.IterationType.NONE)
            .build();

        return MIR.ModelProperties.newBuilder()
            .setMirVersion(MIR.VersionNumber.VERSION_1)
            .setLoopMetadata(cycles)
            .setRecursionMetadata(cycles)
            .setDynamicCollections(false)
            .build();
    }

    private Map<String, MIR.Function> getFunctionMap(SavedBayesNet.Graph graph) {
        Map<String, MIR.Function> functionMap = new HashMap<>();
        functionMap.put(ENTRY_POINT_NAME, getFunctionForGraph(graph));

        return functionMap;
    }

    private MIR.Function getFunctionForGraph(SavedBayesNet.Graph graph) {
        MIR.Function.Builder builder = MIR.Function.newBuilder();

        builder.setName(ENTRY_POINT_NAME);
        builder.getInstructionGroupsBuilder(0)
            .setId(0)
            .setGraph(graph);

        return builder.build();
    }
}
