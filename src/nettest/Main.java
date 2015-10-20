package nettest;

import java.io.File;
import training.*;
import org.json.JSONObject;
import org.apache.commons.io.FileUtils;

/**
 *
 * @author davej
 */
public class Main {

    /**
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        // Load all of the datasets and put them into one 3D double array
        double[][] dataset1 = new double[][]{{1.0}};
        double[][] dataset2 = new double[][]{{1.0}};
        double[][] dataset3 = new double[][]{{1.0}};
        double[][][] datasets = new double[][][]{dataset1,dataset2,dataset3};
        
        // Load the initial neural net into a JSONObject.
        File initialNet = new File("neuralNet.JSON");
        JSONObject json = null;
        try {
            json = new JSONObject(FileUtils.readFileToString(initialNet));
        } catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
        
        // Initialize Trainer(s)
        Trainer geneticAlgorithim = new GeneticAlgorithm();
        
        // Initialize Experimenter
        Experimenter experiment = new Experimenter(json);
        
        // Run experiment
        experiment.run(datasets);
        
    }
    
}
