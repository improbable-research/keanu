//package io.improbable.remorsefulradish.model.bellhop;
//
//import java.io.BufferedReader;
//import java.io.FileReader;
//import java.io.IOException;
//
//import static io.improbable.remorsefulradish.model.util.CommandLineUtils.executeCommand;
//
//public class BellhopRunner {
//    private static final String bellhopFolderPath = "src/main/resources/bellhop/";
//    private String environmentFilePath;
//    private String shadeFilePath;
//    private String bellhopLogFilePath;
//    private static int runCounter = 0;
//
//    public BellhopRunner(String fileName){
//        shadeFilePath = bellhopFolderPath + fileName + ".shd";
//        environmentFilePath = bellhopFolderPath + fileName;
//        bellhopLogFilePath = bellhopFolderPath + fileName + ".prt";
//    }
//
//    public static void resetRunCounter(){
//        runCounter = 0;
//    }
//
//    public static int getRunCount() {
//        return runCounter;
//    }
//
//    public void run(Boolean bellhopDebugLogging){
//        runBellhop(bellhopDebugLogging);
//    }
//
//    public void run(){
//        runBellhop(false);
//    }
//
//    private void runBellhop(Boolean bellhopDebugLogging){
//        runCounter++;
//
//        String commandLineOutput = executeCommand("bellhop.exe " + environmentFilePath, bellhopDebugLogging);
//
//        System.out.format("Ran Bellhop %d times\n", runCounter);
//        if(bellhopDebugLogging){
//            System.out.println(commandLineOutput);
//        }
//        checkBellhopLogFileForErrors(bellhopDebugLogging);
//    }
//
//    public TransmissionLoss readResults(){
//        ShadeFileReader shadeFile = new ShadeFileReader(shadeFilePath);
//        return shadeFile.readTransmissionLoss();
//    }
//
//    private void checkBellhopLogFileForErrors(Boolean bellhopDebugLogging){
//        try{
//            BufferedReader logFileReader = new BufferedReader(new FileReader(bellhopLogFilePath));
//
//            StringBuilder logMessage = new StringBuilder();
//            logFileReader.lines().forEach(line -> logMessage.append(line).append("\n"));
//            if(logMessage.toString().toUpperCase().contains("ERROR")){
//                System.err.print(logMessage.toString());
//            } else if(bellhopDebugLogging){
//                System.out.print(logMessage.toString());
//            }
//
//            logFileReader.close();
//        } catch (IOException e){
//            System.err.print(e);
//        }
//    }
//}
//
//
//
//Collapse 
//
