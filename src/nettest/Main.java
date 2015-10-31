package nettest;

import java.io.File;
import training.*;
import org.json.JSONObject;
import org.apache.commons.io.FileUtils;

public class Main
{
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
    
    // Array of every file name
    public static String[] file = {//                    Index | Description
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
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        int fileIndex = 0;  // Specify the file to use (see file array)
        
        double[][] dataset = DataTools.getDataFromFile(file[fileIndex]);
        
        // Load the initial neural net into a JSONObject.
        File initialNet = new File("neuralNet.JSON");
        JSONObject json = null;
        try {
            json = new JSONObject(FileUtils.readFileToString(initialNet));
        } catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
        
        // Initialize Trainer(s)
        Trainer geneticAlgorithim = new GeneticAlgorithm(GAParams);
        
        // Initialize Experimenter
        Experimenter experiment = new Experimenter(json, new Trainer[]{geneticAlgorithim});
        
        // Run experiment
        experiment.run(dataset);
        
    }
    
}
