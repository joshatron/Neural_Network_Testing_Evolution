package nettest;

import java.io.File;
import training.*;
import org.json.JSONObject;
import org.apache.commons.io.FileUtils;
import feedforward.*;
import java.io.IOException;

public class Main
{
        // Array of every JSON file name. Indices are the same as the datasets.
    public static String[] neuralNetFile = {
        "data/Glass.JSON",//------------------------------> 0
        "data/turkiye-student-evaluation_generic.JSON",//-> 1
        "data/ThoraricSurgery.JSON",//--------------------> 2
        "data/biodeg.JSON",//-----------------------------> 3
        "data/seismic-bumps.JSON",//----------------------> 4
        "data/sensor_readings_24.JSON",//-----------------> 5
        "data/SPECTF.JSON",//-----------------------------> 6
        "data/kr-vs-kp.JSON",//---------------------------> 7
        "data/cmc.JSON",//--------------------------------> 8
        "data/zoo.JSON",//--------------------------------> 9
    };
    
    // Array of every file name
    public static String[] dataFile = {//                    Index | Description
        "data/Glass.csv",//------------------------------> 0   | https://archive.ics.uci.edu/ml/datasets/Glass+Identification
        "data/turkiye-student-evaluation_generic.csv",//-> 1   | https://archive.ics.uci.edu/ml/datasets/Turkiye+Student+Evaluation
        "data/ThoraricSurgery.csv",//--------------------> 2   | https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data
        "data/biodeg.csv",//-----------------------------> 3   | https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation
        "data/seismic-bumps.csv",//----------------------> 4   | https://archive.ics.uci.edu/ml/datasets/seismic-bumps
        "data/sensor_readings_24.csv",//-----------------> 5   | https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data
        "data/SPECTF.csv",//-----------------------------> 6   | https://archive.ics.uci.edu/ml/datasets/SPECTF+Heart
        "data/kr-vs-kp.csv",//---------------------------> 7   | https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
        "data/cmc.csv",//--------------------------------> 8   | https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
        "data/zoo.csv",//--------------------------------> 9   | https://archive.ics.uci.edu/ml/datasets/Zoo
    };
    
    /**
     * [0] => # of children
     * [1] => # of generations
     * [2] => mutation rate
     * [3] => cross over rate
     * [4] => populationSize
     * [5] => # of weights
     * [6] => crossover Type
     */
    public static double[] GAParams = {
            50,
            100,
            0.01,
            0.1,
            50,
            5,
            0
    };
    
    /**
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        int fileIndex = 0;  // Specify the file to use (see file array)
        
        double[][] dataset = DataTools.getDataFromFile(dataFile[fileIndex]);
        
        // Load the initial neural net into a JSONObject.
        File initialNet = new File(neuralNetFile[fileIndex]);
        JSONObject json;
        try {
            json = new JSONObject(FileUtils.readFileToString(initialNet));
        } catch (java.io.IOException e1) {
            FeedForwardNeuralNetwork neuralNet = new FeedForwardNeuralNetwork(1, new int[] { dataset[0].length, 10000, dataset[0].length + 1 }, ActivationFunction.LOGISTIC, ActivationFunction.LOGISTIC);
            try {
                neuralNet.export(new File(neuralNetFile[fileIndex]));
            } catch(IOException e2) {
                System.err.println(e2.getMessage());
            }
            
            json = neuralNet.export();
        }
        
        // Initialize Trainer(s)
        Trainer geneticAlgorithim = new GeneticAlgorithm(GAParams);
        
        // Initialize Experimenter
        Experimenter experiment = new Experimenter(json, new Trainer[]{geneticAlgorithim});
        
        // Run experiment
        experiment.run(dataset);
        
    }
    
}
